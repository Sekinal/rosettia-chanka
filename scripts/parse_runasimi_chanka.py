"""Parse RunaSimi.de's pan-Quechua TSV, extract (Spanish, Ayakuchu/Chanka) pairs.

Source: https://www.runasimi.de/runasimi.txt (ISO-8859-1, 36 tab-separated columns)
  Col 1: Hanan Runasimi (Cuzco-orthography lemma)
  Col 2: Ima simi (POS)
  Col 8: Ayakuchu (the Chanka column — our target)
  Col 29+: Spanish (Español), English (English), etc.
"""
import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path


def download(url: str, dest: str):
    if not Path(dest).exists():
        print(f"Downloading {url} → {dest}")
        urllib.request.urlretrieve(url, dest)
    return dest


def looks_chanka(s: str) -> bool:
    """Chanka has no o/e, no apostrophes, no aspirated stops (kh/qh)."""
    if not s:
        return False
    s_low = s.lower().strip()
    # Reject if Spanish accents
    if re.search(r"[áéíóú]", s_low):
        return False
    # Chanka shouldn't have these glottalized/aspirated patterns
    if re.search(r"[kpqts]'", s_low):
        return False
    if re.search(r"\b(kh|qh|ph|th|chh)[aeiou]", s_low):
        return False
    return True


def normalize_es(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-url", default="https://www.runasimi.de/runasimi.txt")
    ap.add_argument("--cache", default="/tmp/runasimi.txt")
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    download(args.tsv_url, args.cache)

    # Read with ISO-8859-1 since the file is Latin-1
    with open(args.cache, encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    header = rows[0]
    print(f"columns ({len(header)}):")
    for i, h in enumerate(header):
        print(f"  [{i}] {h!r}")

    # Find col indices for: lemma, POS, Ayakuchu (Chanka), Español
    # Be tolerant about header variations.
    def find_col(needles: list[str]) -> int:
        for needle in needles:
            for i, h in enumerate(header):
                if h.strip().lower() == needle.lower():
                    return i
        # fallback substring
        for needle in needles:
            for i, h in enumerate(header):
                if needle.lower() in h.strip().lower():
                    return i
        return -1

    lemma_col = find_col(["Hanan Runasimi", "Runasimi"])
    pos_col = find_col(["Ima simi"])
    ayak_col = find_col(["Ayakuchu", "Ayacucho"])
    es_col = find_col(["Español", "Espanol"])
    print(f"\nlemma col = {lemma_col}  POS col = {pos_col}  Ayakuchu col = {ayak_col}  Español col = {es_col}")
    assert ayak_col >= 0 and es_col >= 0, "Could not locate Ayakuchu or Español columns"

    pairs = []
    n_no_es = 0
    n_no_chanka_in_col = 0
    n_used_lemma_fallback = 0
    for row in rows[1:]:
        if len(row) <= max(ayak_col, es_col):
            continue
        chanka_raw = row[ayak_col].strip()
        es_raw = row[es_col].strip()
        lemma_raw = row[lemma_col].strip() if lemma_col >= 0 else ""

        if not es_raw:
            n_no_es += 1
            continue

        # Pick a Chanka form. If the Ayakuchu column is empty, fall back to the lemma
        # *if* the lemma already looks Chanka (no apostrophes / aspirates).
        if chanka_raw:
            chanka = chanka_raw
        elif lemma_raw and looks_chanka(lemma_raw):
            chanka = lemma_raw
            n_used_lemma_fallback += 1
        else:
            n_no_chanka_in_col += 1
            continue

        # The Chanka cell may have multiple options separated by `;` or `,`
        chanka_opts = []
        for opt in re.split(r"[;,/]", chanka):
            opt = opt.strip()
            if not opt or not looks_chanka(opt):
                continue
            chanka_opts.append(opt)
        if not chanka_opts:
            continue

        # The Spanish cell may have multiple senses; keep them as a tuple but emit one row per (es_lemma_sense, chanka_opt) combo
        # is too expansive. Just take the first Spanish form as the primary lemma.
        es_main = re.split(r"[;,]", es_raw)[0].strip().lower()
        # Drop trailing parens
        es_main = re.sub(r"\s*\([^)]*\)", "", es_main).strip()
        if not es_main or len(es_main) < 2:
            continue

        confidence = "high" if chanka_raw else "lemma_fallback"
        for c in chanka_opts:
            pairs.append((es_main, c, confidence))

    # Dedupe (preserve highest-confidence record per (es, c) pair)
    seen = {}
    for es, c, conf in pairs:
        key = (normalize_es(es), c.lower())
        if key in seen and seen[key][2] == "high":
            continue  # already have high-conf version
        seen[key] = (es, c, conf)
    uniq = list(seen.values())
    n_high = sum(1 for _, _, conf in uniq if conf == "high")
    n_fb = len(uniq) - n_high
    print(f"  high-conf (explicit Ayakuchu): {n_high}")
    print(f"  lemma-fallback:                {n_fb}")

    print(f"\nResults:")
    print(f"  no Spanish gloss:  {n_no_es}")
    print(f"  no Chanka cell:    {n_no_chanka_in_col} (lemma fallback used: {n_used_lemma_fallback})")
    print(f"  raw pairs:         {len(pairs)}")
    print(f"  unique pairs:      {len(uniq)}")

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for es, c, conf in uniq:
            f.write(json.dumps({
                "source": es,
                "target": c,
                "source_name": "runasimi_de_ayakuchu" if conf == "high" else "runasimi_de_lemma_fallback",
                "variant": "quy/chanka",
                "confidence": conf,
            }, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.out_jsonl}")

    # Show samples
    print("\nSample HIGH-CONF pairs (explicit Ayakuchu column):")
    for es, c, conf in [p for p in uniq if p[2] == "high"][:15]:
        print(f"  {es:30s} → {c}")
    print("\nSample LEMMA-FALLBACK pairs (Cuzco lemma reused — Chanka-shaped):")
    for es, c, conf in [p for p in uniq if p[2] == "lemma_fallback"][:10]:
        print(f"  {es:30s} → {c}")


if __name__ == "__main__":
    main()
