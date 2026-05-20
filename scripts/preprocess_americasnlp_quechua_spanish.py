"""Download and filter AmericasNLP 2024 Quechua-Spanish data.

The Quechua-Spanish task includes Ayacucho/Southern Quechua (`quy`) data, but it
is still broad shared-task data rather than reviewed Chanka judicial data. Keep
these outputs in a separate broad-adaptation tier.
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

import polars as pl

from preprocess_somosnlp_dirty_parallel import (
    FilterConfig,
    normalize_text,
    quality_score,
    row_flags,
    words,
)


SOURCE_REPO = "AmericasNLP/americasnlp2024"
SOURCE_PATH = "ST1_MachineTranslation/data/quechua-spanish"
BASE_URL = f"https://raw.githubusercontent.com/{SOURCE_REPO}/master/{SOURCE_PATH}"

RAW_DIR = Path("data/raw/github/americasnlp2024/quechua-spanish")
PROCESSED_DIR = Path("data/processed")
REPORT_PATH = Path("data/interim/americasnlp_quechua_spanish_preprocess_report.md")

LINE_FILES = {
    "train_jw300": ("train.es", "train.quy"),
    "dev": ("dev.es", "dev.quy"),
    # The test set has Spanish inputs only in the public repo.
    "dict": ("dict.es", "dict.quy"),
}
TSV_FILES = {
    "extra": "extra.tsv",
    "synthetic": "synthetic.tsv",
}


def download_file(filename: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / filename
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {url} -> {output_path}")
    urllib.request.urlretrieve(url, output_path)
    return output_path


def read_parallel_lines(subset: str, es_path: Path, quy_path: Path) -> list[dict[str, object]]:
    es_lines = es_path.read_text(encoding="utf-8").splitlines()
    quy_lines = quy_path.read_text(encoding="utf-8").splitlines()
    if len(es_lines) != len(quy_lines):
        raise ValueError(f"Mismatched line counts for {subset}: {len(es_lines)} vs {len(quy_lines)}")

    rows = []
    for idx, (es, quy) in enumerate(zip(es_lines, quy_lines, strict=True)):
        rows.append(
            {
                "source_dataset": "americasnlp2024_quechua_spanish",
                "source_subset": subset,
                "source_origin": subset,
                "source_row_id": idx,
                "es": es,
                "quy": quy,
            }
        )
    return rows


def read_tsv(subset: str, path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for idx, row in enumerate(reader):
            if len(row) != 3:
                continue
            origin, es, quy = row
            rows.append(
                {
                    "source_dataset": "americasnlp2024_quechua_spanish",
                    "source_subset": subset,
                    "source_origin": origin,
                    "source_row_id": idx,
                    "es": es,
                    "quy": quy,
                }
            )
    return rows


def load_raw_dataset(raw_dir: Path) -> pl.DataFrame:
    for filename_pair in LINE_FILES.values():
        for filename in filename_pair:
            download_file(filename, raw_dir)
    for filename in TSV_FILES.values():
        download_file(filename, raw_dir)

    rows: list[dict[str, object]] = []
    for subset, (es_filename, quy_filename) in LINE_FILES.items():
        rows.extend(read_parallel_lines(subset, raw_dir / es_filename, raw_dir / quy_filename))
    for subset, filename in TSV_FILES.items():
        rows.extend(read_tsv(subset, raw_dir / filename))
    return pl.DataFrame(rows)


def preprocess(dataset: pl.DataFrame, config: FilterConfig) -> pl.DataFrame:
    rows = []
    for row in dataset.iter_rows(named=True):
        es = normalize_text(row["es"])
        quy = normalize_text(row["quy"])
        flags = row_flags(es, quy, config)
        rows.append(
            {
                "source_dataset": row["source_dataset"],
                "source_subset": row["source_subset"],
                "source_origin": row["source_origin"],
                "source_row_id": row["source_row_id"],
                "es": es,
                "quy": quy,
                "es_chars": len(es),
                "quy_chars": len(quy),
                "es_words": len(words(es)),
                "quy_words": len(words(quy)),
                "length_ratio_quy_es": len(quy) / max(len(es), 1),
                "filter_flags": "|".join(flags),
                "quality_score": quality_score(flags, es, quy),
            }
        )

    scored = pl.DataFrame(rows)
    deduped = scored.unique(subset=["es", "quy"], keep="first", maintain_order=True)
    return deduped.with_columns(
        (pl.col("filter_flags") == "").alias("passes_high_quality_filters"),
        pl.col("source_subset").is_in(["train_jw300", "dict", "extra"]).alias("is_real_training_source"),
        pl.col("source_subset").eq("synthetic").alias("is_synthetic_source"),
        pl.col("source_subset").eq("dev").alias("is_eval_source"),
    )


def write_report(raw: pl.DataFrame, processed: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset_counts = processed.group_by("source_subset").agg(
        pl.len().alias("rows"),
        pl.col("passes_high_quality_filters").sum().alias("high_quality_rows"),
    )
    flag_counts = (
        processed.select(pl.col("filter_flags").str.split("|").alias("flag"))
        .explode("flag")
        .filter(pl.col("flag") != "")
        .group_by("flag")
        .len()
        .sort("len", descending=True)
    )
    hq_real = processed.filter(pl.col("passes_high_quality_filters") & pl.col("is_real_training_source"))
    hq_with_synth = processed.filter(
        pl.col("passes_high_quality_filters")
        & (pl.col("is_real_training_source") | pl.col("is_synthetic_source"))
    )
    eval_rows = processed.filter(pl.col("is_eval_source"))

    lines = [
        "# AmericasNLP Quechua-Spanish Preprocess Report",
        "",
        f"Source: `https://github.com/{SOURCE_REPO}/tree/master/{SOURCE_PATH}`",
        f"Raw aligned rows loaded: {raw.height:,}",
        f"Unique normalized pairs: {processed.height:,}",
        f"High-quality real SFT rows: {hq_real.height:,}",
        f"High-quality real+synthetic SFT rows: {hq_with_synth.height:,}",
        f"Evaluation rows: {eval_rows.height:,}",
        "",
        "## Subset Counts",
        "",
        subset_counts.sort("source_subset").write_csv(None).strip(),
        "",
        "## Filter Flag Counts",
        "",
        flag_counts.write_csv(None).strip() if flag_counts.height else "No filter flags.",
        "",
        "## Caveat",
        "",
        "AmericasNLP Quechua-Spanish is broad shared-task data. It includes `quy` resources, but it is not reviewed Chanka judicial-domain data.",
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
    hq_real = processed.filter(pl.col("passes_high_quality_filters") & pl.col("is_real_training_source"))
    hq_with_synth = processed.filter(
        pl.col("passes_high_quality_filters")
        & (pl.col("is_real_training_source") | pl.col("is_synthetic_source"))
    )
    eval_rows = processed.filter(pl.col("is_eval_source"))

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    processed.write_parquet(args.processed_dir / "americasnlp_quechua_spanish_scored.parquet")
    processed.write_csv(args.processed_dir / "americasnlp_quechua_spanish_scored.csv")
    hq_real.write_parquet(args.processed_dir / "americasnlp_quechua_spanish_high_quality_real_sft.parquet")
    hq_real.write_csv(args.processed_dir / "americasnlp_quechua_spanish_high_quality_real_sft.csv")
    hq_with_synth.write_parquet(
        args.processed_dir / "americasnlp_quechua_spanish_high_quality_with_synthetic_sft.parquet"
    )
    hq_with_synth.write_csv(
        args.processed_dir / "americasnlp_quechua_spanish_high_quality_with_synthetic_sft.csv"
    )
    eval_rows.write_parquet(args.processed_dir / "americasnlp_quechua_spanish_eval.parquet")
    eval_rows.write_csv(args.processed_dir / "americasnlp_quechua_spanish_eval.csv")
    write_report(raw, processed, args.report_path)

    print(f"Raw aligned rows loaded: {raw.height:,}")
    print(f"Unique normalized pairs: {processed.height:,}")
    print(f"High-quality real SFT rows: {hq_real.height:,}")
    print(f"High-quality real+synthetic SFT rows: {hq_with_synth.height:,}")
    print(f"Evaluation rows: {eval_rows.height:,}")
    print(f"Wrote report: {args.report_path}")


if __name__ == "__main__":
    main()
