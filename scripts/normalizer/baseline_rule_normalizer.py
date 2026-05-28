"""Deterministic rule-based Chanka normalizer.

Implements R1-R7 + L0-L3 from the spec as regex / wordlist lookups.
Used as a baseline to compare against the ML normalizer.
"""
import argparse
import json
import re
from pathlib import Path


# -- Locked spec data --------------------------------------------------------

# R4 forbidden-grapheme wordlist (j→? : depends on word)
J_LOOKUP = {
    "jam": "qam", "jucha": "hucha", "jatun": "hatun", "jampi": "hampi",
    "jawa": "hawa", "jawapi": "hawapi", "jina": "hina", "huj": "huk",
    "juq": "huk", "mijuna": "mikuna", "mijuy": "mikuy", "wajay": "waqay",
}
# qu→k before i/e
QU_TO_K = {"quilla": "killa", "quita": "kita", "quiquin": "kikin", "quita": "kita"}
# c→k (broad)
C_TO_K = {"camay": "kamay", "camcha": "kamcha", "cuyay": "kuyay"}
# f→p
F_TO_P = {"fukuy": "pukuy", "fukuna": "pukuna"}
# cc/qu (mission spellings)
CC_LOOKUP = {"ccollque": "qullqi", "ccapacc": "qapaq"}

R4_LOOKUP = {**J_LOOKUP, **QU_TO_K, **C_TO_K, **F_TO_P, **CC_LOOKUP}

# L1b refonologized loans (Spanish → canonical Quechua spelling)
L1B = {
    "vaca": "waka", "toro": "turu", "caballo": "kawallu", "zapato": "sapatu",
    "vino": "winu", "oveja": "uwiha", "botella": "wutilla", "manzana": "mansana",
    "escuela": "iskuyla", "azúcar": "asukar", "jabón": "hawun", "lápiz": "lapis",
}

# L3 toponyms (preserve case)
L3_TOPONYMS = {
    "Cuzco": "Qusqu", "Cusco": "Qusqu", "Ayacucho": "Aya Kuchu",
    "Apurímac": "Apu Rimaq", "Machupicchu": "Machu Pikchu",
    "Huancavelica": "Wanka Willka", "Perú": "Piruw",
    "Andahuaylas": "Anta Waylla", "Cajamarca": "Kasha Marka",
}
# L3 surnames preserved as-is
L3_SURNAMES_PRESERVE = {"Huamán", "Quispe", "Mamani", "Condori", "Yupanqui"}
# §8.5 conservative preserve (proper nouns / acronyms)
PRESERVE = {"Jesus", "Jesús", "Dios", "Jehová", "Jehova", "Cristo", "Espíritu",
            "MINEDU", "UNESCO", "UNSAAC", "Pedro", "María", "Ecuador", "Kennedy",
            "Faulk", "Covid"}

# Spec-required wordlist for `qan→qam` (2sg pronoun)
QAN_PRONOUN = {"qan", "Qan", "QAN"}


def is_quechua_token(tok: str) -> bool:
    """Return True if token is Quechua-shaped (not a Spanish loan / proper noun)."""
    if not tok or not tok[0].isalpha():
        return False
    if tok in PRESERVE or tok in L3_SURNAMES_PRESERVE:
        return False
    if tok in L3_TOPONYMS:
        return False  # Will be handled separately
    if tok in L1B:
        return False  # Will be handled separately
    # Spanish-form proper noun (capitalized, not sentence-initial check needed)
    return True


def normalize_token(tok: str) -> str:
    """Apply R1-R7 to a presumed Quechua token."""
    out = tok
    # Lookup full-token forms first (highest precedence)
    if tok.lower() in R4_LOOKUP:
        return R4_LOOKUP[tok.lower()]
    if tok in L1B:
        return L1B[tok]
    # R1: strip apostrophes (Collao ejective marker)
    out = out.replace("'", "").replace("’", "")
    # R2: de-aspirate chh/kh/ph/qh/th → ch/k/p/q/t
    out = re.sub(r"chh", "ch", out)
    out = re.sub(r"kh", "k", out)
    out = re.sub(r"ph", "p", out)
    out = re.sub(r"qh", "q", out)
    out = re.sub(r"th", "t", out)
    # R3: e→i, o→u (Quechua tokens only)
    out = out.replace("e", "i").replace("o", "u").replace("E", "I").replace("O", "U")
    # R6: ll before q (and not already ll): add l
    out = re.sub(r"(?<!l)l(q)", r"ll\1", out)
    # R7: qan (2sg pronoun) → qam, allim → allin
    if out in ("qan", "Qan"):
        out = out[:-1] + "m"
    if out == "allim":
        out = "allin"
    # R5: diphthong break — minimal set
    out = re.sub(r"^hua", "wa", out)
    out = re.sub(r"^huai", "way", out)
    out = re.sub(r"^huay", "way", out)
    out = re.sub(r"yahuar", "yawar", out)
    out = re.sub(r"ai\b", "ay", out)
    return out


def normalize_sentence(s: str) -> str:
    """Normalize a sentence: tokenize, classify, apply rules per-token."""
    # Replace L3 toponyms (multi-word safe via word-boundary)
    for k, v in L3_TOPONYMS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    # L0: strip nisqa after L1b/L1c loans (heuristic: match <loan>(<suffix>) nisqa(<suffix>))
    s = re.sub(
        r"(Televisor|computadora|Tablet|teléfono|celular|Internet|radio|doctor|Banco|avión|televisor)(\S*)\s+nisqa(\S*)",
        r"\1\2\3", s, flags=re.IGNORECASE
    )
    tokens = re.findall(r"\S+", s)
    out = []
    for t in tokens:
        # Strip surrounding punctuation
        m = re.match(r"^(\W*)(.+?)(\W*)$", t)
        if not m:
            out.append(t)
            continue
        prefix, core, suffix = m.groups()
        if core in PRESERVE or core in L3_SURNAMES_PRESERVE:
            out.append(t)
            continue
        if core in L1B:
            out.append(prefix + L1B[core] + suffix)
            continue
        if not is_quechua_token(core):
            out.append(t)
            continue
        out.append(prefix + normalize_token(core) + suffix)
    return " ".join(out)


# §7 gold list (subset) — same as eval_normalizer_vllm.py
from importlib.util import spec_from_file_location, module_from_spec
import sys
sys.path.insert(0, "/home/ieqr/Desktop/research/rosettia/scripts/normalizer")
spec = spec_from_file_location("evnv", "/home/ieqr/Desktop/research/rosettia/scripts/normalizer/eval_normalizer_vllm.py")
evnv = module_from_spec(spec)
spec.loader.exec_module(evnv)
GOLD = evnv.GOLD_PAIRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    correct = 0
    fails = []
    for src, expected in GOLD:
        got = normalize_sentence(src)
        ok = got.strip() == expected.strip()
        correct += int(ok)
        if not ok:
            fails.append({"input": src, "expected": expected, "got": got})
    acc = correct / len(GOLD)
    print(f"Rule-based baseline on §7 gold list: {correct}/{len(GOLD)} = {acc*100:.1f}%")
    print(f"\n=== Failures ({len(fails)}) ===")
    for f in fails[:20]:
        print(f"  in:  {f['input']}")
        print(f"  exp: {f['expected']}")
        print(f"  got: {f['got']}")
        print()
    if args.out_json:
        json.dump({"correct": correct, "total": len(GOLD), "acc": acc, "fails": fails},
                  open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
