"""Parse Benito Zuasnabar 2018 'Diccionario Básico Quechua Chanka' (diccionary.pdf).

Strategy: pypdf preserves line structure, so we parse line-by-line.
  - Pages 4-9 are QU→ES: `lemma POS. spanish_gloss.` (may wrap to next line until '.')
  - Pages 10-16 are ES→QU: `spanish_lemma chanka_word [; chanka_alt]` (one entry per visual line; alt seps `;`, `/`)
"""
import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader


POS_PATTERN = r"(?:adj|adv|conec|conec\.por eso|dem|interj|num|part|posp|pron|sust|verb|v\.tr|v\.int|interj\.afirm)"

SECTION_LETTERS = {"A", "B", "C", "Ch", "D", "E", "F", "G", "H", "I", "J", "K", "L", "Ll",
                   "M", "N", "Ñ", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y", "Z"}


def extract_lines_by_page(pdf_path: str) -> dict[int, list[str]]:
    r = PdfReader(pdf_path)
    out = {}
    for i, page in enumerate(r.pages, 1):
        txt = page.extract_text() or ""
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        out[i] = lines
    return out


def is_section_header(line: str) -> bool:
    return line in SECTION_LETTERS


def is_header_or_pagenum(line: str) -> bool:
    if re.fullmatch(r"\d+", line):
        return True
    if line in ("QUECHUA–CASTELLANO", "QUECHUA-CASTELLANO", "CASTELLANO–QUECHUA", "CASTELLANO-QUECHUA"):
        return True
    return False


# ---------- QU→ES parsing ----------

def collect_qu_es_entries(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Group continuation lines into single entries terminated by '.'. Returns [(quechua_headword, [spanish_glosses])]."""
    # Concatenate lines; an entry ends at '.' that's NOT inside an abbreviation like "adj."
    buf = ""
    entries = []
    for ln in lines:
        if is_section_header(ln) or is_header_or_pagenum(ln):
            continue
        if buf:
            buf += " " + ln
        else:
            buf = ln
        # Heuristic: an entry ends when the buffer contains exactly one POS+gloss pair OR
        # when the buffer ends with '.' AND the next char would start a new lowercase lemma.
        # Simpler: try to parse the buffer; if successful, emit and reset.
        m = parse_qu_es_buf(buf)
        if m is not None:
            entries.append(m)
            buf = ""
    if buf:
        m = parse_qu_es_buf(buf)
        if m is not None:
            entries.append(m)
    return entries


def parse_qu_es_buf(buf: str) -> tuple[str, list[str]] | None:
    """Try to parse `<quechua_lemma[/alt]> <POS>. <gloss>.` from buf. Return None if buf is mid-entry."""
    m = re.match(
        r"^([A-Za-zñÑ'][A-Za-zñÑ' /\(\)0-9]*?)\s+"
        r"(" + POS_PATTERN + r"(?:[\./](?:" + POS_PATTERN.replace("(?:", "(?:") + r"))?\.)\s+"
        r"(.+?)\.\s*$",
        buf,
    )
    if not m:
        return None
    qu = m.group(1).strip()
    gloss = m.group(3).strip()
    qu_first = qu.split("/")[0].split("(")[0].strip()
    spanish_glosses = []
    for g in re.split(r"[;,]", gloss):
        g = g.strip().lower()
        # Drop trailing parenthetical clarifier
        g = re.sub(r"\s*\([^)]*\)", "", g).strip()
        if not g or len(g) < 2:
            continue
        spanish_glosses.append(g)
    if not spanish_glosses:
        return None
    return (qu_first, spanish_glosses)


# ---------- ES→QU parsing ----------

def collect_es_qu_entries(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Each line is one entry: `<spanish_lemma> <chanka_word[; alt; ...]>`. Returns [(es, [chanka_options])]."""
    pairs = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if is_section_header(ln) or is_header_or_pagenum(ln):
            i += 1
            continue
        # Merge with continuation lines if this line ends mid-word (e.g., hyphenation or wrapped headword)
        # In practice, ES→QU entries are short and fit one line; wrap rarely needed.
        es, chankas = split_es_qu_line(ln)
        if es and chankas:
            pairs.append((es, chankas))
        i += 1
    return pairs


def looks_like_chanka(tok: str) -> bool:
    """Heuristic for whether a token is Chanka (vs Spanish)."""
    t = tok.lower().strip(".,;:/()")
    if not t:
        return False
    # Spanish accents -> not Chanka
    if re.search(r"[áéíóú]", t):
        return False
    # Common Spanish endings
    if t.endswith(("ado", "ido", "mente", "ión", "ar", "er", "ir")) and len(t) > 4:
        return False
    # Chanka features
    return any(c in t for c in "qwy") or (len(t) >= 3 and t[-1] in "aiu")


def split_es_qu_line(line: str) -> tuple[str, list[str]]:
    """Split a line like `abandonar saqiy` or `alegrar (a otro) kusichiy` or `así hina; chayna` into (es, [chanka])."""
    # Handle multi-Spanish-token headword via `(...)` clarifier
    # Use regex to find the first Chanka-looking token; everything before is ES.
    toks = line.split()
    if len(toks) < 2:
        return ("", [])
    # Find split point: the first index `i >= 1` where toks[i] looks Chanka AND toks[i-1] doesn't end with `(` or comma
    split_idx = None
    paren_depth = 0
    for i, t in enumerate(toks):
        paren_depth += t.count("(") - t.count(")")
        if i == 0:
            continue
        if paren_depth > 0:
            continue
        if looks_like_chanka(t):
            split_idx = i
            break
    if split_idx is None:
        # Fallback: split after first token
        split_idx = 1
    es = " ".join(toks[:split_idx]).lower().strip()
    chanka_text = " ".join(toks[split_idx:])
    # Strip parens from ES
    es_clean = re.sub(r"\s*\([^)]*\)\s*", " ", es).strip()
    # Sometimes the parens contain useful disambiguation; keep as separate entries.
    # Split chanka by `;`, `/`, or commas
    chanka_opts = []
    for c in re.split(r"[;/]", chanka_text):
        c = c.strip().lower()
        c = re.sub(r"[.,;:]+$", "", c)
        if not c:
            continue
        # Take just the first word if multiple (drop trailing noise like POS tags that leaked in)
        first_word = c.split()[0]
        if looks_like_chanka(first_word):
            chanka_opts.append(first_word)
    return (es_clean, chanka_opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="/home/ieqr/Downloads/diccionary.pdf")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    lines_by_page = extract_lines_by_page(args.pdf)
    print(f"Loaded {len(lines_by_page)} pages")

    qu_es_lines = []
    for p in range(4, 10):
        qu_es_lines.extend(lines_by_page.get(p, []))
    es_qu_lines = []
    for p in range(10, 17):
        es_qu_lines.extend(lines_by_page.get(p, []))

    qu_es = collect_qu_es_entries(qu_es_lines)
    es_qu = collect_es_qu_entries(es_qu_lines)
    print(f"QU→ES: {len(qu_es)} headwords")
    print(f"ES→QU: {len(es_qu)} headwords")

    # Build es → set(chanka) merge from both directions
    merged: dict[str, set[str]] = {}
    for qu, spanish_list in qu_es:
        for sp in spanish_list:
            merged.setdefault(sp, set()).add(qu)
    for es, chanka_list in es_qu:
        for c in chanka_list:
            merged.setdefault(es, set()).add(c)

    out = {
        "metadata": {
            "source": "Benito Zuasnabar 2018, Diccionario Básico Quechua Chanka",
            "variety": "Chanka (Ayacucho-Huancavelica)",
            "n_qu_es_headwords": len(qu_es),
            "n_es_qu_headwords": len(es_qu),
            "n_es_lemmas_merged": len(merged),
            "n_total_pairs": sum(len(v) for v in merged.values()),
        },
        "es_to_chanka": {k: sorted(v) for k, v in sorted(merged.items())},
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {len(merged)} Spanish lemmas, {out['metadata']['n_total_pairs']} pairs to {args.out_json}")

    with open(args.out_jsonl, "w") as f:
        for es, qus in sorted(merged.items()):
            for qu in sorted(qus):
                f.write(json.dumps({
                    "source": es,
                    "target": qu,
                    "source_name": "benito_2018_chanka_dict",
                    "variant": "quy/chanka",
                }, ensure_ascii=False) + "\n")
    print(f"Wrote parallel JSONL to {args.out_jsonl}")


if __name__ == "__main__":
    main()
