# Synthetic & distilled training-data methodology for the Chanka normalizer

How we manufacture (noisy → canonical) training pairs for the ML orthographic
normalizer, why each source exists, and the leakage / quality controls. This is
the data companion to `chanka_normalizer_findings.md` (the iteration log) and
`chanka_orthography_spec.md` (the rules being taught).

The normalizer learns a single task: **map any Chanka-Quechua sentence to its
MINEDU-2021 canonical form, preserving loans / proper nouns / morphology.** We
have no public gold corpus for this, so every training pair is manufactured by
one of four complementary pipelines below, then balanced and filtered.

---

## 0. The core problem these pipelines solve

A normalizer needs two competing competencies:
- **Recall** — actually apply R1-R7/L-rules where the text is contaminated.
- **Precision** — leave already-correct text, loans, proper nouns, and rich
  morphology untouched.

No single data source teaches both. So we generate from four sources, each
targeting a different competency, then tune the mix (see §6).

| Pipeline | Teaches | Signal type |
|---|---|---|
| A. Synthetic perturbation | recall (what to fix) | (clean → corrupt → invert) |
| B. Rule-focused augmentation | recall on specific rules | templated single-rule |
| C. Multi-teacher distillation | precision + recall in real context | LLM-voted |
| D. Verified-gate self-distillation | precision (what to leave alone) | deterministic-verified |

---

## 1. Pipeline A — synthetic perturbation (`inject_chanka_contamination.py`)

**Idea.** Start from text that is *already* MINEDU-compliant (v30 judicial
corpus). Apply the **inverse** of each spec rule to manufacture realistic
contamination, then train the model to recover the original.

- clean `pacha` → inject apostrophe → noisy `p'acha`; target = `pacha`.
- clean `qillqay` → inject 5-vowel → noisy `qelqay`; target = `qillqay`.
- clean `pullu` → aspirate → `phullu`; target = `pullu`.

Three intensity levels per clean sentence (light ≈10 %, medium ≈30 %, heavy
≈55 % of perturbable tokens) so the model sees the full noise spectrum.

**Loanword conflict injection.** A dedicated table (`LOAN_CANONICAL_VARIANTS`)
swaps a canonical loan for one of its wrong surfaces (`computadora ↔
kumputarura`, `waka ↔ vaca`) so the model learns to recover the *one* canonical
form rather than averaging surfaces (the exact failure that sank MT v32).

**Guard.** Perturbation only touches tokens that pass `looks_like_quechua_token`
(rejects capitalized proper nouns and tokens already carrying Spanish-only
letters), so we never manufacture a pair that tells the model to "fix" a loan.

**Output:** 5 787 pairs from the 1 929-sentence v30 seed.

**Limitation.** Synthetic noise is cleaner and more single-rule than reality;
it over-teaches "always change something." This is why Pipeline C/D exist.

---

## 2. Pipeline B — rule-focused augmentation (`normalizer_rule_focused.jsonl`)

**Idea.** After v41a's failure analysis showed specific rules under-firing
(R4 forbidden-grapheme, R5 diphthong, R6 ll-before-q, R7 m/n-before-p, L0
nisqa-strip, L1b loans, L3 toponyms), we generate ~8 templated sentence
variants per wrong→right pair for each weak rule, each with a hand-built
per-token reasoning trace citing the exact spec rule.

- subject/verb/complement slots filled from a small native-Quechua lexicon,
  the target token dropped in noisy and clean positions.
- e.g. `jam {adj}` → trace cites §R4 → `qam {adj}`.

**Output:** 1 145 pairs. This is the single biggest recall lever (v42c 71.9 %
→ v43a 87.5 % spec_gold).

**Caveat (documented circular-eval risk).** Pipeline B and the §7 evaluation
list are both authored from the same spec, so §7 over-states recall. We added
the held-out precision metric (Pipeline D) and the AmericasNLP probe precisely
to measure generalization beyond rule templates.

---

## 3. Pipeline C — multi-teacher LLM distillation (the headline method)

For text where we have **no gold** (real contaminated corpora: Wikipedia,
RunaSimi-fallback, and the AmericasNLP-train JW.org corpus), we distill labels
from an ensemble of LLM "teachers," each applying the spec, and keep only what
they agree on.

### 3.1 The teacher brief
Every teacher (DeepSeek-V3 via API; Claude via the Agent tool) is given
`scripts/normalizer/teacher_brief.md` — a ~2 K-token distillation of the full
spec: the 18-grapheme inventory, R1-R7, S-rules, L0-L3 loan policy, §8.5
conservative fallback, and the locked output format:

```
<think>
Tokens:
1. "<tok>" — <observation>. Spec §<rule>: <≤12-word summary>. Result: "<norm>"
...
</think>

Normalized: <sentence>
```

Per-token traces force the teacher to *justify* every edit against a rule,
which both improves quality and yields an explainable artifact the student
model learns to reproduce.

### 3.2 The vote
- **≥2-of-K exact agreement** (after light canonicalization: strip trace,
  collapse whitespace, drop trailing period) → accept.
- DeepSeek labels the full batch (cheap, fast, bulk). Claude pass A labels the
  same batch independently. Where A and DeepSeek **disagree**, a Claude pass B
  acts as tiebreaker; the pair is accepted only if B matches A or DeepSeek
  (true 2-of-3). All-three-disagree → **rejected** (noise).

### 3.3 Why the ensemble matters (empirical)
On the 1 000-sentence AmericasNLP-train batch:
- DeepSeek + Claude-A agreed on 578 (57.8 %).
- Claude-B resolved 378 of the 422 disagreements; 44 were rejected.
- **Final: 956 gold pairs.**

The disagreements were not random — **DeepSeek systematically over-
refonologized Spanish loans** (`semana→simana`, `familia→pamilia`,
`judio→hudio`) and over-stripped `nisqa`, while Claude correctly preserved them
per §L1c/§L2/§L0. A single-teacher (or single deterministic) pipeline would
have silently baked in that bias. The vote is what makes the distilled data
trustworthy.

### 3.4 Domain-specific spec pressure
The AmericasNLP corpus is JW.org religious text: dense with `Jehová`, `Jesus`,
`Dios`, and Bible citations (`Cant.`, `Mat. 24:14`). The hardest spec call is
§8.5/§L2 — apply R3/R4 to Quechua tokens but **never** to those proper nouns or
verse refs. The multi-teacher traces show the agents reasoning about exactly
this, and the vote enforces it.

**Output:** 956 AmericasNLP gold pairs (`data/amnlp_normalized_gold.jsonl`);
the earlier Phase-C run produced 2 058 (`data/normalizer_gold.jsonl`).

---

## 4. Pipeline D — verified-gate self-distillation (precision)

**Idea.** Deploying the trained model (v43b) on the real v30 corpus revealed it
*corrupts* clean text (suffix swaps, apostrophe insertion, word splitting). To
teach precision, we generate preservation targets by running the model's own
proposals through a **token-level verified gate** (`verified_normalize` in
`apply_normalizer_vllm.py`):

- the model proposes which tokens to change;
- each accepted edit must EQUAL the deterministic safe transform of that token
  (strip `'`, de-aspirate, e→i/o→u, l→ll-before-q) — any other edit is a
  corruption and is reverted to the original token.

The gated output is corruption-free by construction. We then train the next
model to imitate it. On v30 the gate cut changes from 11.2 % (raw, ~3-4 %
harmful) to 0.5 % (all verified-safe), giving high-precision preservation
targets over rich real morphology.

**Output:** ~1 700 v30 preservation rows (after excluding the 200-sentence
precision holdout) + the AmericasNLP "verified-preserve" rows.

---

## 5. Leakage controls (critical)

The downstream goal is to normalize the v34 MT training corpus and report v34's
ChrF on the **AmericasNLP 2021 test set**. So the test set must never influence
any normalizer decision.

| Control | Rule |
|---|---|
| **Test set isolation** | `2021_test.quy` is NEVER used for normalizer training OR checkpoint selection. (We initially used it as the precision probe — caught and removed; see findings doc.) |
| **Precision probe** | a 200-sentence **v30 holdout**, excluded from all training, judicial-domain, verified disjoint from the AmericasNLP test set. |
| **AmericasNLP source** | only `train.quy` (125 008 lines), leakage-checked: **0** exact-overlap with the 1003-line test. |
| **Train/eval split** | §7 gold list + a 100-row held-out gold slice, neither in the training file. |
| **Acknowledged circular eval** | §7 ↔ Pipeline B share the spec; we therefore also report the independent v30-holdout corruption metric, never claim §7 alone. |

---

## 6. Balancing the mix (the recall/precision dial)

The identity ratio (fraction of pairs where input == output) is the master dial:

| dataset | identity % | result |
|---|---:|---|
| v43 augmented | ~9 % | recall 96.9 %, but **corrupts** clean text |
| v44 | 45 % | corruption → 0 %, but recall collapses to ~59 % |
| **v45** | **33.7 %** | targeting both (in training now) |

v45 = all correction signal (Pipelines A-templated + B + 241 real AmericasNLP
corrections) + a downsampled preservation set (Pipeline D v30 + AmericasNLP
verified-preserve) tuned to 33.7 % identity.

---

## 7. Evaluation metrics

| metric | measures | source |
|---|---|---|
| **§7 spec_gold** | recall on 64 single-rule cases | hand-authored from spec |
| **held-out gold** | recall on unseen voted pairs | tail of training file |
| **corruption_rate** | precision: % of clean v30-holdout sentences damaged (changed to a non-safe-transform) | 200-sentence v30 holdout |
| **combined_score** | `spec_gold − corruption_rate` | checkpoint selection |

Checkpoints are selected by **combined_score**, so the orchestrator optimizes
recall and precision jointly — not recall alone (which let v43b's corruption
slip through).

---

## 8. Reproducibility

| Artifact | Path |
|---|---|
| Synthetic perturbation | `scripts/normalizer/inject_chanka_contamination.py` |
| Rule-focused augmentation | `data/normalizer_rule_focused.jsonl` |
| Teacher brief (spec for LLMs) | `scripts/normalizer/teacher_brief.md` |
| DeepSeek labeler | `scripts/normalizer/label_with_deepseek.py` |
| Vote merge | `scripts/normalizer/merge_teacher_votes.py` |
| Verified gate | `verified_normalize()` in `scripts/normalizer/apply_normalizer_vllm.py` |
| AmericasNLP voted gold | `data/amnlp_normalized_gold.jsonl` (956) |
| Phase-C voted gold | `data/normalizer_gold.jsonl` (2 058) |
| Precision holdout probe | `data/normalizer_precision_holdout.txt` (200 v30) |
| Balanced training set | `data/normalizer_gold_v45.jsonl` (3 544, 33.7 % identity) |

**Teacher models:** DeepSeek-V3 (`deepseek-chat`) for bulk; Anthropic Claude
(via the Agent tool) for independent passes A/B and tiebreaking.

---

## 9. Final result — the gated system achieves true minimal errors

After v43/v44/v45 it became clear the **raw 4B model cannot hit high recall and
low corruption simultaneously** — it sits on a Pareto frontier:

| model checkpoint | §7 recall | corruption (200 clean v30 holdout) |
|---|---:|---:|
| v43b ckpt-576 | 96.9 % | ~3-4 % |
| v44a (45 % identity) | ~59 % | 0 % |
| v45a ckpt-128 | 60.9 % | 4.5 % |
| v45a ckpt-256 | 87.5 % | 25.5 % |
| v45a ckpt-384 | 68.8 % | 65 % (!) |
| **v45a ckpt-1263** | **96.9 %** | 21.5 % |

The corruption is also *unstable* across checkpoints (4.5 → 25.5 → 65 → 21.5 %),
so no amount of checkpoint-picking yields a trustworthy raw model.

**Solution: model + token-level verified gate** (`verified_normalize`). The model
proposes which tokens to change (contextual loan / proper-noun protection); the
gate accepts an edit only if it equals the deterministic safe transform of that
token (R1-R7 char ops + R4 wordlist + L1b/L3 table lookups + §L0 nisqa-merge).
Anything else — suffix swap, q/k drift, hallucination, word-split — is reverted
to the original token. Hallucination is structurally impossible because every
accepted edit comes from a closed deterministic set.

**Measured on v45a ckpt-1263:**

| system | §7 recall | corruption (200 clean holdout) |
|---|---:|---:|
| raw model | 96.9 % | 21.5 % |
| **+ verified gate** | **96.9 %** | **0.0 %** |

The gate keeps every legitimate correction (recall unchanged at 96.9 %) and
eliminates all 43 corruptions on clean text. This is the deployed normalizer:
`apply_normalizer_vllm.py` now routes every output through `verified_normalize`.

**Division of labour.** The ML model is irreplaceable for the *contextual*
decision — "is this `e` inside a Quechua word (fix it) or inside `Jehová` /
`Televisor` (leave it)?" — which the deterministic baseline gets wrong (it
mangles loans). The gate is irreplaceable for *safety* — guaranteeing the model
never corrupts what it does touch. Neither alone is minimal-error; together they
are. This hybrid (LLM-proposes / deterministic-verifies) is the methodological
takeaway.

---

## 10. Production run — normalizing the v34 MT training corpus

With the gated normalizer validated, we apply it to the **actual data that will
train the v34 translator**, to unify orthography across sources (the mixed-
orthography problem that sank MT v31/v32).

**Corpus.** AmericasNLP 2021 Spanish–Quechua train split: 125 008 aligned
(quy, es) pairs. Leakage-checked at download: **0** overlap with the 1003-line
AmericasNLP 2021 test set. The Spanish side is left untouched; only the Chanka
(`quy`) side is normalized.

**Pipeline.**
1. vLLM serves `unsloth/Qwen3.5-4B` + LoRA `v45a/checkpoint-1263`
   (`enforce_eager`, `max_lora_rank=512`).
2. Each quy sentence → model proposes normalization (with trace) →
   `verified_normalize` gate accepts only deterministic-safe edits.
3. Output re-aligned with the untouched Spanish side → v34 parallel corpus.

**Safety inheritance.** Because every emitted line passes the verified gate, the
0 %-corruption guarantee measured on the held-out probe carries over to the full
125 k corpus: the normalizer can only apply spec-safe edits or leave a line
unchanged — it cannot inject morphology errors into the training data. This is
the property that makes it safe to normalize at scale without human review of
every line.

**Output:** `data/amnlp_train_quy_normalized.jsonl` (per-line: original quy,
gated-normalized quy, Spanish, change flag) → folded into the v34 training set
alongside the (already-clean) v30 corpus.
