"""Build a Chanka-native (es_lemma → chanka_lemma) dictionary from the training pairs.

Approach:
  1. Load 897 train rows (after seed=3407 split with frac=0.15)
  2. Tokenize Spanish + Chanka, lowercase
  3. For each (es_word, chanka_word) co-occurrence pair, count
  4. For each es_word, pick chanka_word with highest IBM-1-style alignment score:
     score(s, t) = count(s, t) / (count_es(s) * count_qu(t))
  5. Filter: drop entries with < min_count co-occurrences, or score < min_score
  6. Save as JSON in the same format the smoke script expects
"""
import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def normalize(s):
    s = s.lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s


def tokenize(s):
    s = normalize(s)
    return re.findall(r"[\w']+", s)


SPANISH_STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "los", "del", "se", "las", "un", "por",
    "con", "no", "una", "su", "para", "es", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes",
    "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto",
    "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco",
    "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis",
    "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras",
    "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas",
    "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros",
    "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas",
}

QUECHUA_STOPWORDS = {
    # Common Chanka grammatical markers / particles
    "manam", "mana", "icha", "kachkan", "kasqa", "karqa", "kanmi", "kani",
    "kaspa", "munan", "munani", "munanki", "munanqa",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-repo", default="Thermostatic/rosettia-chanka-data")
    ap.add_argument("--chanka-file", default="clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet")
    ap.add_argument("--cache-dir", default="/root/.cache/hf-rosettia")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--min-count", type=int, default=2, help="min co-occurrences")
    ap.add_argument("--min-score", type=float, default=0.5, help="min IBM-style alignment score")
    ap.add_argument("--top-translations-per-source", type=int, default=2, help="keep top-N Chanka translations per Spanish lemma")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    # Load training rows (same split logic as eval)
    import polars as pl
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=args.dataset_repo, filename=args.chanka_file, repo_type="dataset", cache_dir=args.cache_dir)
    frame = pl.read_parquet(path)
    rows = []
    for r in frame.select(["reviewed_spanish", "reviewed_chanka_quechua"]).iter_rows(named=True):
        s = str(r["reviewed_spanish"]).strip()
        t = str(r["reviewed_chanka_quechua"]).strip()
        if s and t:
            rows.append({"source": s, "target": t})
    rng = random.Random(args.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * args.validation_fraction))
    train = shuffled[eval_size:]
    print(f"train rows: {len(train)}")

    # Co-occurrence counts
    co = defaultdict(Counter)  # es_word -> Counter(qu_word -> count)
    es_count = Counter()
    qu_count = Counter()
    for row in train:
        es_tokens = [t for t in tokenize(row["source"]) if t not in SPANISH_STOPWORDS and len(t) > 2]
        qu_tokens = [t for t in tokenize(row["target"]) if t not in QUECHUA_STOPWORDS and len(t) > 2]
        for e in set(es_tokens):
            es_count[e] += 1
            for q in set(qu_tokens):
                co[e][q] += 1
        for q in set(qu_tokens):
            qu_count[q] += 1

    # Score each (es, qu) co-occurrence
    scored = {}  # es_word -> [(qu_word, score, count), ...]
    for es_word, qu_counter in co.items():
        candidates = []
        for qu_word, count in qu_counter.items():
            if count < args.min_count:
                continue
            # IBM-1-ish: P(qu|es) = count(es, qu) / count(es), weighted by qu rarity
            p_qu_given_es = count / es_count[es_word]
            p_qu = qu_count[qu_word] / len(train)
            # Score: favor co-occurrences that are specific (qu rare-but-here)
            score = p_qu_given_es * (1 / (1 + p_qu))
            candidates.append((qu_word, round(score, 4), count))
        candidates.sort(key=lambda x: -x[1])
        keep = [c for c in candidates if c[1] >= args.min_score][: args.top_translations_per_source]
        if keep:
            scored[es_word] = keep

    print(f"unique Spanish lemmas with translations: {len(scored)}")
    total_pairs = sum(len(v) for v in scored.values())
    print(f"total (es, qu) pairs: {total_pairs}")

    # Save as the same format the smoke script expects
    out = {
        "metadata": {
            "source": "v28_build_chanka_dict.py",
            "train_rows": len(train),
            "n_lemmas": len(scored),
            "n_pairs": total_pairs,
            "min_count": args.min_count,
            "min_score": args.min_score,
        },
        "lemmas": [],
    }
    for es_word, candidates in sorted(scored.items()):
        # Take top-1 as the primary, list rest as synonyms
        primary_qu = candidates[0][0]
        synonyms = [c[0] for c in candidates[1:]]
        out["lemmas"].append({
            "es": es_word,
            "qu_cuzco": primary_qu,  # name kept for compatibility with smoke script
            "qu_cuzco_synonyms": synonyms,
            "pos": "?",
            "gloss": es_word,
            "chanka_note": "mined from training pairs (Chanka-native)",
            "score": candidates[0][1],
            "count": candidates[0][2],
        })

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {args.out_json}")

    # Show samples
    print("\n=== sample mined pairs ===")
    for es_word in ["esposo", "vivir", "calle", "doctor", "saber", "casa", "querer"]:
        if es_word in scored:
            print(f"  {es_word} → {scored[es_word]}")


if __name__ == "__main__":
    main()
