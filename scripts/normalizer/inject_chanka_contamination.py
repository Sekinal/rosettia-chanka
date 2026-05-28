"""Inject Cuzco/missionary contamination into MINEDU-compliant Chanka text.

The output is a corpus of (noisy_quy, clean_quy) pairs suitable for training
a seq2seq orthographic normalizer.  Every perturbation is the INVERSE of a
rule documented in `docs/findings/chanka_orthography_spec.md`.

We DO NOT perturb tokens that look Spanish (capitalized proper nouns or
tokens containing forbidden Chanka letters that would otherwise indicate
a loan, e.g. 'e', 'o', 'f', 'b', 'd', 'g', 'v'): those are presumed loans
or proper nouns and the spec says to leave them alone.

Three intensity levels: light (≈10% of perturbable tokens), medium (≈30%),
heavy (≈55%).  Each level is emitted as a separate output row so the
student model sees the full noise spectrum.
"""
import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path


# Inverse rule tables (clean → noisy candidates).
# Each entry is a list — perturbation picks uniformly at random.

# R1 inverse: glottalize plain stops (only some realistic environments).
# A plain stop followed by a vowel inside a word becomes "stop+'" with low p.
GLOTTAL_CANDIDATES = {"k": "k'", "q": "q'", "p": "p'", "t": "t'", "ch": "ch'"}

# R2 inverse: aspirate plain stops.
ASPIR_CANDIDATES = {"k": "kh", "q": "qh", "p": "ph", "t": "th", "ch": "chh"}

# R3 inverse: i→e, u→o adjacent to uvular q (where the [e]/[o] allophone is natural).
# Pattern: replace 'i' with 'e' / 'u' with 'o' when adjacent to q.

# R4 inverse: write forbidden-graphemes for canonical ones.
J_FOR_H = {"hampi": "jampi", "hatun": "jatun", "hawa": "jawa", "hucha": "jucha",
           "hina": "jina", "hap": "jap", "huk": "huj"}
C_FOR_K = {"k": "c"}  # very common in missionary spelling (camay, cusi, etc.)
QU_FOR_K = {"k": "qu"}  # only before i/e (quilla = killa)
F_FOR_P = {"p": "f"}  # rare loan spelling

# R5 inverse: collapse w/y to Spanish-style diphthongs.
DIPHTHONG_INVERSE = [
    (r"\bwa", "hua"), (r"\bwi", "hue"), (r"\bwu", "huo"),
    (r"\bya", "ia"), (r"\bye", "ie"),
    (r"ay\b", "ai"), (r"aw\b", "au"), (r"iy\b", "ei"),
]

# R6 inverse: drop one l from llq.
LL_BEFORE_Q_INVERSE = [(r"ll(q)", r"l\1")]

# R7 inverse: swap m/n before p.
M_BEFORE_P_INVERSE = [(r"m(p)", r"n\1")]  # mp -> np (wrong in roots)
QAM_INVERSE = [(r"\bqam\b", "qan")]
ALLIN_INVERSE = [(r"\ballin\b", "allim")]

# Suffix inverse: -nchik → -nchis/-nchix; -yki contractions.
SUFFIX_INVERSE = [
    (r"nchik\b", "nchis"),
    (r"nchik\b", "nchix"),
    (r"yki\b", "ki"),
    (r"iyki\b", "iki"),
]

# Loanword form conflicts: train the normalizer to map ANY surface to the canonical.
# Each entry: canonical (per spec §L1b/L1c) → variant surfaces that should be REJECTED.
# Random perturbation may substitute the canonical with one of the variants in the noisy
# version, teaching the model to recover the canonical form.

LOAN_CANONICAL_VARIANTS = {
    # Category (b) — refonologized form is canonical; reject Spanish-form variants
    "waka":     ["vaca", "wakas"],
    "turu":     ["toro"],
    "kawallu":  ["caballo", "kaballo"],
    "sapatu":   ["zapatu", "zapato"],
    "uwiha":    ["uwija", "oveja"],
    "wutilla":  ["botella", "wotella"],
    "winu":     ["vino"],
    "lapis":    ["lapiz", "lápiz"],
    "iskuyla":  ["escuela", "eskuyla"],
    "asukar":   ["asúcar", "azukar", "azúcar"],
    "hawun":    ["jabón", "jawón", "jawun"],
    "mansana":  ["manzana"],
    "kamisa":   ["camisa"],
    "siwara":   ["cebada"],
    "kawra":    ["cabra", "kabra"],
    "wallpa":   ["gallina"],
    "riru":     ["dedo"],
    "kuwarta":  ["cuarta"],
    # Category (c) — Spanish form is canonical; reject refonologized variants
    "computadora": ["kumputarura", "kumputatura", "computatora"],
    "televisor":   ["tiliwisur", "tilibisur"],
    "teléfono":    ["tilipunu", "tilifunu", "telefono"],
    "celular":     ["silular", "selular"],
    "internet":    ["internét", "internit"],
    "radio":       ["raryu", "radyu"],
    "barco":       ["warku", "bargu"],
    "avión":       ["awyón", "awyun"],
    "carro":       ["karru"],
    "gobierno":    ["gubirnu", "kuwirnu"],
    "ministerio":  ["ministiryu"],
    "banco":       ["wanku"],
    "presidente":  ["prisidinti"],
    "doctor":      ["ruktur"],
    "Pedro":       ["Pidru"],
    "Ecuador":     ["Ikwadur"],
}

# Compound merge: split compounds back together (e.g., "yachay wasi" → "yachaywasi").
COMPOUND_MERGE = [
    ("yachay wasi", "yachaywasi"),
    ("Pacha Mama", "Pachamama"),
    ("Tayta Inti", "Taytainti"),
    ("Aya Kuchu", "Ayakuchu"),
    ("Apu Rimaq", "Apurimaq"),
    ("Machu Pikchu", "Machupikchu"),
    ("Wanka Willka", "Wankawillka"),
    ("Wira Qucha", "Wiraqucha"),
]

# Stand-alone token rewrites that are common Cuzco/missionary forms.
DIRECT_REWRITES = {
    "ñuqa": ["ñoqa", "noqa", "ñuga"],
    "kachkani": ["kashani"],          # -chka → -sha (Cuzco)
    "kachkan": ["kashan"],
    "kachkanchik": ["kashanchis", "kashanchix"],
    "kachkanku": ["kashanku"],
    "punchaw": ["p'unchaw", "punchau"],
    "kimsa": ["kinsa"],
    "haykap": ["hayk'aq", "haykaq"],
    "pichqa": ["pisqa", "pishqa"],
    "achka": ["ashka", "askha"],
    "tukuy": ["tukoy"],
    "sunqu": ["sonqo"],
    "qillqa": ["qellqa"],
    "qillqay": ["qellqay"],
    "qichwa": ["qechwa"],
    "qusqu": ["qosqo"],
    "qipa": ["qhepa", "qepa"],
    "qullqi": ["qollqe", "qolqe"],
    "unquy": ["onqoy"],
    "lluqsiy": ["lloqsiy"],
    "qillu": ["q'illu", "q'ello"],
    "kuska": ["kushka", "kuchka"],
    "wikuña": ["wik'uña"],
    "minka": ["mink'a"],
    "llamkay": ["llamk'ay"],
    "machay": ["mach'ay"],
    "hucha": ["jucha"],
    "hatun": ["jatun", "qatun", "atun"],
    "hampi": ["jampi", "qampi"],
    "hawa": ["jawa", "qawa"],
    "hawapi": ["jawapi", "qawapi"],
    "killa": ["quilla"],
    "kamay": ["camay"],
    "mikuy": ["mikhuy"],
    "kuchi": ["khuchi"],
    "pukuy": ["phukuy", "fukuy"],
    "qari": ["qhari"],
    "qaway": ["qhaway"],
    "chala": ["chhalla"],
    "pacha": ["p'acha"],
    "tanta": ["t'anta"],
    "maytu": ["mayt'u"],
    "sacha": ["sach'a"],
    "tika": ["t'ika"],
    "kaspi": ["k'aspi"],
    "tampa": ["t'ampa", "t'anpa"],
    "muti": ["mut'i"],
    "wata": ["wat'a"],
    "hapiy": ["hap'iy"],
    "qapiy": ["q'apiy"],
    "hichay": ["hich'ay"],
    "patma": ["phatma"],
    "tuñiy": ["thuñiy"],
    "llantu": ["llanthu", "llant'u"],
    "qincha": ["qhincha"],
    "qaytu": ["q'aytu"],
    "qumir": ["q'umir"],
    "ichu": ["ichhu"],
    "aqa": ["aqha"],
    "ñaqa": ["ñaqha"],
    "rapi": ["raphi", "laphi", "rap'i"],
    "sapi": ["saphi"],
    "qachun": ["qhachun"],
    "kuyapayay": ["khuyapayay"],
    "upakuy": ["uphakuy"],
    "waltay": ["walthay"],
    "rikchaq": ["rikch'aq"],
    "tiqtiy": ["tiqt'iy", "tiqthiy"],
    "qullqip": ["qullqiq"],            # Chanka genitive -pa, Cuzco -q
    "wallqa": ["walqa"],
    "sallqa": ["salqa", "salq'a"],
    "allqu": ["alqu", "allku"],
    "lliklla": ["lliqlla"],
    "mikuna": ["mihuna", "mijuna"],
    "wayta": ["huaita"],
    "wayna": ["huaina"],
    "yawar": ["yahuar"],
    "wawqiy": ["huauqei"],
    "wasi": ["wasy"],
    "uchuy": ["huch'uy"],
    "aku": ["hak'u"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def looks_like_quechua_token(tok: str) -> bool:
    """Return True if the token should be considered Quechua (eligible for perturbation).

    Reject if:
      - empty / pure punctuation / pure digit
      - contains uppercase initial AND length>2 (likely proper noun/loan;
        we still want to perturb sentence-initial Quechua words though, so
        be lenient on length 1-2 like single-letter abbreviations)
      - contains a letter that is FORBIDDEN in Chanka (b, d, e, f, g, o, v,
        x, z, j) — those signal a Spanish loan, proper noun, or non-clean
        input already; we don't add more noise on top
    """
    if not tok:
        return False
    if all(not c.isalpha() for c in tok):
        return False
    if tok[0].isupper() and len(tok) > 2 and not tok.isupper():
        # capitalized non-acronym: likely proper noun
        return False
    low = tok.lower()
    if any(c in low for c in "bdefgjovxz"):
        return False
    return True


def quechua_words_in_sentence(s: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, token)] for every Quechua-like word in s.

    Boundaries are computed on `re.finditer(r"\\w+", s, re.UNICODE)`; for
    each match, `looks_like_quechua_token` filters out probable loans.
    """
    out = []
    for m in re.finditer(r"\w+", s, re.UNICODE):
        tok = m.group(0)
        if looks_like_quechua_token(tok):
            out.append((m.start(), m.end(), tok))
    return out


def apply_token_rewrite(tok: str, rng: random.Random, p: float) -> str:
    """Apply one or more inverse rules to a single Quechua token."""
    low = tok.lower()

    # 1. Direct lexical inverse first (these are highest-precision)
    if low in DIRECT_REWRITES and rng.random() < p:
        candidates = DIRECT_REWRITES[low]
        cand = rng.choice(candidates)
        # restore capitalization
        if tok[0].isupper():
            cand = cand[0].upper() + cand[1:]
        return cand

    out = low

    # 2. Aspirate or glottalize a plain stop (mutually exclusive per token)
    if rng.random() < p * 0.4:
        # pick first occurrence of a perturbable stop
        for stop in ["ch", "k", "q", "p", "t"]:
            idx = out.find(stop)
            # only inside the word, never word-final
            if 0 <= idx < len(out) - len(stop):
                if rng.random() < 0.5:
                    repl = ASPIR_CANDIDATES[stop]
                else:
                    repl = GLOTTAL_CANDIDATES[stop]
                out = out[:idx] + repl + out[idx + len(stop):]
                break

    # 3. Vowel: i→e or u→o near q (allophonic environment)
    if rng.random() < p * 0.3:
        # i adjacent to q
        m = re.search(r"(q[ih]?)i", out)
        if m and rng.random() < 0.5:
            out = out[:m.end() - 1] + "e" + out[m.end():]
        m2 = re.search(r"i(q)", out)
        if m2 and rng.random() < 0.5:
            out = out[:m2.start()] + "e" + out[m2.start() + 1:]
        m3 = re.search(r"(q[ih]?)u", out)
        if m3 and rng.random() < 0.5:
            out = out[:m3.end() - 1] + "o" + out[m3.end():]
        m4 = re.search(r"u(q)", out)
        if m4 and rng.random() < 0.5:
            out = out[:m4.start()] + "o" + out[m4.start() + 1:]

    # 4. Substitute h → j (common missionary spelling)
    if rng.random() < p * 0.15 and "h" in out and not out.startswith("ch"):
        if rng.random() < 0.5:
            out = out.replace("h", "j", 1)

    # restore capitalization
    if tok[0].isupper() and out:
        out = out[0].upper() + out[1:]
    return out


def sentence_level_perturb(s: str, rng: random.Random, p: float) -> str:
    """Apply sentence-level inverses: diphthong-collapse, suffix-spacing, compound-merge."""
    out = s

    # Loanword variant substitution (replace canonical form with a rejected variant)
    # Token-boundary aware so we don't touch substrings.
    for canonical, variants in LOAN_CANONICAL_VARIANTS.items():
        if rng.random() < p * 0.6:
            pat = re.compile(r"\b" + re.escape(canonical) + r"\b")
            if pat.search(out):
                out = pat.sub(rng.choice(variants), out, count=1)

    # Compound merge (e.g., "yachay wasi" → "yachaywasi")
    for clean, noisy in COMPOUND_MERGE:
        if rng.random() < p * 0.3 and clean in out:
            out = out.replace(clean, noisy, 1)

    # ll → l before q
    if rng.random() < p * 0.25:
        for pat, rep in LL_BEFORE_Q_INVERSE:
            out = re.sub(pat, rep, out, count=1)

    # m → n before p in roots (occasional)
    if rng.random() < p * 0.15:
        for pat, rep in M_BEFORE_P_INVERSE:
            out = re.sub(pat, rep, out, count=1)

    # qam → qan
    if rng.random() < p * 0.3:
        out = re.sub(r"\bqam\b", "qan", out, count=1, flags=re.IGNORECASE)
        out = re.sub(r"\bQam\b", "Qan", out, count=1)

    # diphthong collapse (rare)
    if rng.random() < p * 0.2:
        pat, rep = rng.choice(DIPHTHONG_INVERSE)
        out = re.sub(pat, rep, out, count=1)

    # suffix forms
    if rng.random() < p * 0.25:
        pat, rep = rng.choice(SUFFIX_INVERSE)
        out = re.sub(pat, rep, out, count=1)

    return out


def perturb_sentence(s: str, level: str, rng: random.Random) -> str:
    """Apply token-level + sentence-level perturbation at the given intensity."""
    p = {"light": 0.10, "medium": 0.30, "heavy": 0.55}[level]

    # token-level
    spans = quechua_words_in_sentence(s)
    if spans:
        # walk right-to-left so we can splice without shifting offsets
        out = s
        for start, end, tok in reversed(spans):
            new_tok = apply_token_rewrite(tok, rng, p)
            if new_tok != tok:
                out = out[:start] + new_tok + out[end:]
    else:
        out = s

    # sentence-level
    out = sentence_level_perturb(out, rng, p)
    return out


def normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="CSV/JSONL file of MINEDU-compliant Chanka sentences (column 'reviewed_chanka_quechua' or one sentence per line).")
    ap.add_argument("--out-jsonl", required=True,
                    help="JSONL: {noisy, clean, level} per row.")
    ap.add_argument("--per-sentence", type=int, default=3,
                    help="Number of noisy variants per clean sentence (one each of light/medium/heavy by default).")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load sentences
    sentences: list[str] = []
    if args.input.endswith(".csv"):
        import csv
        with open(args.input) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # pick first non-empty cell
                for v in row.values():
                    if v and v.strip() and v.strip() != "reviewed_chanka_quechua":
                        sentences.append(normalize_unicode(v))
                        break
    elif args.input.endswith(".jsonl"):
        with open(args.input) as f:
            for line in f:
                d = json.loads(line)
                for key in ("clean", "chanka", "reviewed_chanka_quechua", "target", "quy"):
                    if key in d:
                        sentences.append(normalize_unicode(d[key]))
                        break
    else:
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(normalize_unicode(line))

    print(f"Loaded {len(sentences)} clean sentences", file=sys.stderr)

    # Generate noisy variants
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    levels = ["light", "medium", "heavy"]
    written = 0
    with open(args.out_jsonl, "w") as f:
        for s in sentences:
            for level in levels[: args.per_sentence]:
                # try up to 4 times to get a noisy version that differs from clean
                for _ in range(4):
                    noisy = perturb_sentence(s, level, rng)
                    if noisy != s:
                        break
                f.write(json.dumps({"noisy": noisy, "clean": s, "level": level}, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} pairs to {args.out_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()
