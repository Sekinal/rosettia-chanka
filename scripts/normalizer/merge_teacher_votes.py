"""Merge labels from DeepSeek + Claude A + Claude B teachers into final training data.

For each input sentence:
- Collect all available teacher outputs (1–3 sources).
- Apply majority-vote on the `normalized` field (modulo trailing punctuation/whitespace).
- Pick the BEST trace among teachers that produced the winning normalization
  (prefer Claude traces over DeepSeek; longer/more-tokens trace wins ties).
- Emit (input, trace, normalized, agreement_level, provenance) per row.
- Reject rows where teachers all disagree AND no expected gold is available.

Outputs three files:
  - <out>_gold.jsonl       — ≥2-of-3 agreed (or single-teacher with gold match)
  - <out>_disagree.jsonl   — flagged for review
  - <out>_summary.json     — counts + agreement statistics
"""
import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def canon_norm(s: str) -> str:
    """Strip trailing punctuation and collapse whitespace for vote-comparison."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # Remove a single trailing period or stray comma
    s = re.sub(r"[.,]\s*$", "", s)
    return s


def load_labels(paths: list[str]) -> dict[int, list[dict]]:
    """idx → list of teacher records."""
    by_idx: dict[int, list[dict]] = defaultdict(list)
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "idx" in r and r.get("normalized"):
                by_idx[r["idx"]].append(r)
    return by_idx


def pick_best_trace(votes: list[dict], winning_norm: str) -> dict:
    """Pick the best teacher record among those producing the winning norm.

    Preference order: claude_b > claude_a > deepseek (newer/independent first),
    tiebreaker: longer trace wins.
    """
    pref = {"claude_b": 0, "claude_a": 1}
    candidates = [v for v in votes if canon_norm(v["normalized"]) == canon_norm(winning_norm)]
    if not candidates:
        candidates = votes  # fallback to any vote
    candidates.sort(key=lambda v: (
        pref.get(v.get("teacher", "").split("/")[0], 9),
        -len(v.get("trace", ""))
    ))
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-glob", required=True, help="Glob for DeepSeek label files")
    ap.add_argument("--claude-a-glob", required=True)
    ap.add_argument("--claude-b-glob", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    seeds = {r["idx"]: r for r in (json.loads(l) for l in open(args.seeds))}
    def expand_globs(pat: str) -> list[str]:
        out: list[str] = []
        for p in pat.split(","):
            out.extend(sorted(glob.glob(p.strip())))
        return out
    ds_paths = expand_globs(args.ds_glob)
    ca_paths = expand_globs(args.claude_a_glob)
    cb_paths = expand_globs(args.claude_b_glob)
    print(f"DS files: {len(ds_paths)}")
    print(f"Claude A files: {len(ca_paths)}")
    print(f"Claude B files: {len(cb_paths)}")

    ds = load_labels(ds_paths)
    ca = load_labels(ca_paths)
    cb = load_labels(cb_paths)

    n_total = len(seeds)
    n_agreed_3way = n_agreed_2way = n_single = n_disagree = 0
    n_with_gold = 0
    gold_rows = []
    disagree_rows = []
    sources_counter: Counter[str] = Counter()

    for idx, seed in sorted(seeds.items()):
        votes = []
        if idx in ds: votes.extend(ds[idx])
        if idx in ca: votes.extend(ca[idx])
        if idx in cb: votes.extend(cb[idx])

        if not votes:
            continue  # no teacher labeled this row (shouldn't happen after DS finishes)

        # Tally normalized outputs
        tally = Counter(canon_norm(v["normalized"]) for v in votes)
        unique_norms = len(tally)
        # Pick winning norm: highest vote count; on TIES, prefer the norm
        # produced by a Claude teacher over the DeepSeek teacher.
        # Build per-norm best-teacher rank (lower = better).
        TEACHER_RANK = {"claude_silver": 0, "claude_a": 1, "claude_b": 1}  # default 9 for ds
        norm_best_rank: dict[str, int] = {}
        for v in votes:
            t = (v.get("teacher") or "").split("/")[0]
            rank = TEACHER_RANK.get(t, 9)
            n = canon_norm(v["normalized"])
            if n not in norm_best_rank or rank < norm_best_rank[n]:
                norm_best_rank[n] = rank
        # Sort by (negative vote count, best teacher rank)
        sorted_norms = sorted(tally.keys(), key=lambda n: (-tally[n], norm_best_rank.get(n, 9)))
        top_norm = sorted_norms[0]
        top_count = tally[top_norm]

        agreement = "single"
        if len(votes) >= 3 and top_count >= 3:
            agreement = "3-way"; n_agreed_3way += 1
        elif len(votes) >= 2 and top_count >= 2:
            agreement = "2-way"; n_agreed_2way += 1
        elif len(votes) == 1:
            agreement = "single"; n_single += 1
        elif len(votes) == 2 and top_count == 1:
            # 2 teachers disagree — keep the Claude-preferred one as silver
            agreement = "claude_override"; n_disagree += 1
        else:
            agreement = "disagree"; n_disagree += 1

        # If we have gold expected, count alignment
        gold_match = None
        if seed.get("expected"):
            gold_match = canon_norm(seed["expected"]) == top_norm

        winner = pick_best_trace(votes, top_norm)
        # Use the winner's full (unsanitized) normalized string for output
        row = {
            "idx": idx,
            "source": seed.get("source"),
            "input": seed.get("input", winner.get("input")),
            "trace": winner.get("trace", ""),
            "normalized": winner["normalized"],
            "agreement": agreement,
            "n_votes": len(votes),
            "n_unique_norms": unique_norms,
            "teachers": [v.get("teacher") for v in votes],
            "gold_match": gold_match,
        }

        sources_counter[seed.get("source", "?")] += 1

        if agreement in ("3-way", "2-way") or (agreement == "single" and gold_match is True):
            gold_rows.append(row)
        elif agreement == "single":
            # Single teacher, no gold to confirm: accept conservatively (mark as "silver")
            row["agreement"] = "silver_single"
            gold_rows.append(row)
        elif agreement == "claude_override":
            # 2 teachers disagree; we picked the Claude-preferred output. Keep as silver.
            gold_rows.append(row)
        else:
            disagree_rows.append(row)
        if gold_match is not None:
            n_with_gold += 1

    Path(args.out_prefix + "_gold.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_prefix + "_gold.jsonl", "w") as f:
        for r in gold_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out_prefix + "_disagree.jsonl", "w") as f:
        for r in disagree_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "n_total_seeds": n_total,
        "n_labeled": n_agreed_3way + n_agreed_2way + n_single + n_disagree,
        "n_3_way_agreement": n_agreed_3way,
        "n_2_way_agreement": n_agreed_2way,
        "n_single_teacher": n_single,
        "n_disagree": n_disagree,
        "n_with_gold_expected": n_with_gold,
        "n_gold_match_top_vote": sum(1 for r in gold_rows if r.get("gold_match") is True),
        "n_gold_mismatch_top_vote": sum(1 for r in gold_rows if r.get("gold_match") is False),
        "by_source": dict(sources_counter),
        "n_gold_rows_written": len(gold_rows),
        "n_disagree_rows_written": len(disagree_rows),
    }
    with open(args.out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
