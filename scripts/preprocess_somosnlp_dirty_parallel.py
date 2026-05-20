"""Download and filter the SomosNLP Spanish-Quechua dirty corpus.

This corpus is useful as broad translation SFT data, not as clean Chanka data.
The output keeps provenance and filter metadata so later training code can mix it
separately from the reviewed Chanka judicial corpus.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import polars as pl


DATASET_ID = "somosnlp-hackathon-2022/spanish-to-quechua"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/data"
SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

RAW_DIR = Path("data/raw/huggingface/somosnlp_spanish_to_quechua")
PROCESSED_DIR = Path("data/processed")
REPORT_PATH = Path("data/interim/somosnlp_dirty_preprocess_report.md")


SPANISH_STOPWORDS = {
    "de",
    "del",
    "la",
    "las",
    "el",
    "los",
    "que",
    "por",
    "para",
    "con",
    "una",
    "uno",
    "un",
    "como",
    "en",
    "se",
    "su",
    "sus",
    "es",
    "son",
    "al",
    "lo",
    "no",
    "si",
}

QUECHUA_HINTS = {
    "qa",
    "mi",
    "m",
    "ta",
    "pa",
    "pi",
    "paq",
    "wan",
    "kuna",
    "man",
    "manta",
    "chik",
    "chu",
    "hina",
    "hinaspa",
    "nispa",
    "kan",
    "kasqa",
    "runa",
    "qichwa",
    "quechua",
}

BRACKETED_METADATA_RE = re.compile(r"^\s*[\[\(].{0,80}[\]\)]\s*$")
ONLY_PUNCT_OR_NUMBERS_RE = re.compile(r"^[\W\d_]+$", re.UNICODE)
WORD_RE = re.compile(r"[\wáéíóúüñÁÉÍÓÚÜÑ-]+", re.UNICODE)


@dataclass(frozen=True)
class FilterConfig:
    min_chars: int = 18
    max_chars: int = 700
    min_words: int = 3
    max_length_ratio: float = 3.2
    min_length_ratio: float = 0.32
    max_char_overlap: float = 0.72


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"([¿¡])\s+", r"\1", text)
    return text


def words(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def char_overlap(left: str, right: str) -> float:
    left_chars = {c for c in left.lower() if c.isalpha()}
    right_chars = {c for c in right.lower() if c.isalpha()}
    if not left_chars or not right_chars:
        return 0.0
    return len(left_chars & right_chars) / max(len(left_chars), len(right_chars))


def token_overlap(left_tokens: list[str], right_tokens: list[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    left = set(left_tokens)
    right = set(right_tokens)
    return len(left & right) / max(len(left), len(right))


def has_spanish_signal(tokens: list[str]) -> bool:
    return any(token in SPANISH_STOPWORDS for token in tokens)


def has_quechua_signal(tokens: list[str]) -> bool:
    if any(token in QUECHUA_HINTS for token in tokens):
        return True
    return any(
        token.endswith(("nku", "nchik", "ykichik", "sqa", "sun", "chis", "manta", "man", "paq"))
        for token in tokens
    )


def row_flags(es: str, qu: str, config: FilterConfig) -> list[str]:
    es_tokens = words(es)
    qu_tokens = words(qu)
    flags: list[str] = []

    if not es or not qu:
        flags.append("empty_side")
    if len(es) < config.min_chars or len(qu) < config.min_chars:
        flags.append("too_short")
    if len(es) > config.max_chars or len(qu) > config.max_chars:
        flags.append("too_long")
    if len(es_tokens) < config.min_words or len(qu_tokens) < config.min_words:
        flags.append("too_few_words")
    if ONLY_PUNCT_OR_NUMBERS_RE.fullmatch(es) or ONLY_PUNCT_OR_NUMBERS_RE.fullmatch(qu):
        flags.append("punct_or_numbers_only")
    if BRACKETED_METADATA_RE.fullmatch(es) or BRACKETED_METADATA_RE.fullmatch(qu):
        flags.append("bracketed_metadata")

    ratio = len(qu) / max(len(es), 1)
    if ratio < config.min_length_ratio or ratio > config.max_length_ratio:
        flags.append("extreme_length_ratio")

    es_norm = es.casefold()
    qu_norm = qu.casefold()
    if es_norm == qu_norm:
        flags.append("exact_copy")
    if token_overlap(es_tokens, qu_tokens) > 0.62 and char_overlap(es, qu) > config.max_char_overlap:
        flags.append("high_overlap_copy_risk")

    if not has_spanish_signal(es_tokens):
        flags.append("weak_spanish_signal")
    if not has_quechua_signal(qu_tokens):
        flags.append("weak_quechua_signal")

    return flags


def quality_score(flags: list[str], es: str, qu: str) -> float:
    score = 1.0
    hard_penalties = {
        "empty_side": 1.0,
        "exact_copy": 1.0,
        "punct_or_numbers_only": 1.0,
        "bracketed_metadata": 0.8,
        "too_short": 0.5,
        "too_few_words": 0.4,
        "too_long": 0.25,
        "extreme_length_ratio": 0.35,
        "high_overlap_copy_risk": 0.5,
        "weak_spanish_signal": 0.12,
        "weak_quechua_signal": 0.2,
    }
    for flag in flags:
        score -= hard_penalties.get(flag, 0.1)

    ratio = len(qu) / max(len(es), 1)
    if 0.65 <= ratio <= 1.75:
        score += 0.06
    if es.endswith((".", "?", "!", ":", "”", '"')) and qu.endswith((".", "?", "!", ":", "”", '"')):
        score += 0.04
    return max(0.0, min(1.0, score))


def download_split(split: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = SPLIT_FILES[split]
    output_path = output_dir / filename
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {url} -> {output_path}")
    urllib.request.urlretrieve(url, output_path)
    return output_path


def load_raw_dataset(raw_dir: Path) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for split in SPLIT_FILES:
        path = download_split(split, raw_dir)
        frame = pl.read_parquet(path).with_columns(pl.lit(split).alias("split"))
        frames.append(frame)
    dataset = pl.concat(frames, how="vertical_relaxed")
    if not {"es", "qu"}.issubset(dataset.columns):
        raise ValueError(f"Expected columns 'es' and 'qu', got: {dataset.columns}")
    return dataset


def preprocess(dataset: pl.DataFrame, config: FilterConfig) -> pl.DataFrame:
    rows = []
    for idx, row in enumerate(dataset.iter_rows(named=True)):
        es = normalize_text(row["es"])
        qu = normalize_text(row["qu"])
        flags = row_flags(es, qu, config)
        rows.append(
            {
                "source_dataset": DATASET_ID,
                "source_split": row["split"],
                "source_row_id": idx,
                "es": es,
                "qu": qu,
                "es_chars": len(es),
                "qu_chars": len(qu),
                "es_words": len(words(es)),
                "qu_words": len(words(qu)),
                "length_ratio_qu_es": len(qu) / max(len(es), 1),
                "filter_flags": "|".join(flags),
                "quality_score": quality_score(flags, es, qu),
            }
        )

    scored = pl.DataFrame(rows)
    deduped = scored.unique(subset=["es", "qu"], keep="first", maintain_order=True)
    return deduped.with_columns(
        (pl.col("filter_flags") == "").alias("passes_high_quality_filters"),
        (
            (pl.col("quality_score") >= 0.72)
            & ~pl.col("filter_flags").str.contains("empty_side|exact_copy|punct_or_numbers_only|bracketed_metadata")
        ).alias("passes_broad_sft_filters"),
    )


def write_report(raw: pl.DataFrame, processed: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_counts = processed.group_by("source_split").agg(
        pl.len().alias("rows"),
        pl.col("passes_high_quality_filters").sum().alias("high_quality_sft_rows"),
        pl.col("passes_broad_sft_filters").sum().alias("broad_sft_rows"),
    )
    flag_counts = (
        processed.select(pl.col("filter_flags").str.split("|").alias("flag"))
        .explode("flag")
        .filter(pl.col("flag") != "")
        .group_by("flag")
        .len()
        .sort("len", descending=True)
    )

    lines = [
        "# SomosNLP Dirty Corpus Preprocess Report",
        "",
        f"Source dataset: `{DATASET_ID}`",
        f"Raw rows: {raw.height:,}",
        f"Unique normalized pairs: {processed.height:,}",
        f"High-quality SFT rows: {processed.filter(pl.col('passes_high_quality_filters')).height:,}",
        f"Broad backup SFT rows: {processed.filter(pl.col('passes_broad_sft_filters')).height:,}",
        "",
        "## Split Counts",
        "",
        split_counts.sort("source_split").write_csv(None).strip(),
        "",
        "## Filter Flag Counts",
        "",
        flag_counts.write_csv(None).strip() if flag_counts.height else "No filter flags.",
        "",
        "## Caveat",
        "",
        "This is broad, dirty Spanish-Quechua data. Keep it separate from reviewed Chanka data and use it only for the broad SFT stage.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--min-chars", type=int, default=FilterConfig.min_chars)
    parser.add_argument("--max-chars", type=int, default=FilterConfig.max_chars)
    parser.add_argument("--min-words", type=int, default=FilterConfig.min_words)
    parser.add_argument("--min-length-ratio", type=float, default=FilterConfig.min_length_ratio)
    parser.add_argument("--max-length-ratio", type=float, default=FilterConfig.max_length_ratio)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FilterConfig(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_words=args.min_words,
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
    )

    raw = load_raw_dataset(args.raw_dir)
    processed = preprocess(raw, config)
    high_quality_sft = processed.filter(pl.col("passes_high_quality_filters"))
    broad_sft = processed.filter(pl.col("passes_broad_sft_filters"))

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    processed.write_parquet(args.processed_dir / "somosnlp_spanish_to_quechua_scored.parquet")
    processed.write_csv(args.processed_dir / "somosnlp_spanish_to_quechua_scored.csv")
    high_quality_sft.write_parquet(args.processed_dir / "somosnlp_spanish_to_quechua_high_quality_sft.parquet")
    high_quality_sft.write_csv(args.processed_dir / "somosnlp_spanish_to_quechua_high_quality_sft.csv")
    broad_sft.write_parquet(args.processed_dir / "somosnlp_spanish_to_quechua_broad_sft.parquet")
    broad_sft.write_csv(args.processed_dir / "somosnlp_spanish_to_quechua_broad_sft.csv")
    write_report(raw, processed, args.report_path)

    print(f"Raw rows: {raw.height:,}")
    print(f"Unique normalized pairs: {processed.height:,}")
    print(f"High-quality SFT rows: {high_quality_sft.height:,}")
    print(f"Broad backup SFT rows: {broad_sft.height:,}")
    print(f"Wrote report: {args.report_path}")


if __name__ == "__main__":
    main()
