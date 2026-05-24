# DeepSeekMath-V2 Translation Adaptation

Source paper: `/home/ieqr/Downloads/2511.22570v1.pdf`, **DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning**.

## Core Transfer

DeepSeekMath-V2 is useful for RosettIA because it stops treating RL as final-answer matching. The paper trains a verifier, then a generator that must both solve the task and evaluate its own solution. For Chanka translation, the analogous target is not "show math proof steps"; it is "produce a faithful Chanka translation and correctly identify translation flaws."

Mapping:

- Math problem `X` -> Spanish source sentence.
- Proof `Y` -> Chanka translation candidate.
- Proof verifier -> translation verifier that scores faithfulness, Chanka grammar, terminology, entity preservation, Spanish leakage, repetition, and formatting.
- Meta-verifier -> verifier-analysis judge that checks whether the verifier's stated issues are real rather than hallucinated or hidden.
- Self-verifying proof generator -> translator that emits final translation plus calibrated self-analysis and score.
- Scaled proof verification -> K-sample candidate translation search with multiple verifier/self-verifier analyses to surface subtle issues.

The intended output format for the generator is:

```text
Traduccion final: <solo la traduccion al quechua chanka>
Autoevaluacion: <errores detectados o declaracion de que no ve errores>
Puntaje: \boxed{<0.0 a 1.0>}
```

External metrics must strip everything except `Traduccion final`. The self-analysis is training-time scaffolding, not part of the translation delivered to users.

## Implemented

`scripts/train_gspo_chanka_unsloth.py` now has reward profile `self_verifiable_translation_2511`.

Behavior:

- Enables the structured self-verification prompt.
- Parses `Traduccion final`, `Autoevaluacion`, and `Puntaje: \boxed{...}`.
- Scores only the final translation against the clean reference for chrF++/BLEU/token overlap/length/copy/leakage guards.
- Adds a self-evaluation reward that compares the model's boxed self-score to the reference-aware translation score.
- Adds a meta-analysis proxy that penalizes false confidence, missing issue acknowledgement on weak translations, and exaggerated self-doubt on strong translations.
- Uses the DeepSeekMath-V2 generator weighting idea: `0.76 * translation_quality + 0.24 * calibrated_self_evaluation`.
- Keeps existing overlong, source-copy, Spanish leakage, repetition, and chat-artifact penalties.
- During final eval, structured outputs are stripped down to final translations before corpus metrics are computed.
- Patches Qwen chat templates during GSPO so TRL's internal prompt rendering closes the `<think>` block. Without this, Qwen starts completions inside a reasoning trace and all generations clip at the completion limit.

`scripts/build_self_verifiable_translation_data.py` builds cold-start JSONL data from the clean Chanka corpus:

- `translation_verifier_cold_start.jsonl`: candidate translation scoring examples, using existing clean references plus synthetic corruptions.
- `translation_meta_verifier_cold_start.jsonl`: faithful and deliberately flawed verifier analyses for meta-verifier training.
- `self_verifiable_generator_sft.jsonl`: generator examples in the final/self-analysis/boxed-score format.
- `self_verifiable_thinking_generator_sft.jsonl`: generator examples with a bounded `Analisis de traduccion` field before the final translation.

The thinking variant now uses explicit translation primitive tags:

- `[SIGNIFICADO]`: meaning, omissions, additions, and faithfulness.
- `[GRAMATICA]`: Chanka grammar, suffixes, case/possession, verb shape, and naturalness.
- `[ENTIDADES]`: names, numbers, people, places, and dates.
- `[TERMINOLOGIA]`: glossary terms and stable equivalents.
- `[ANTI_COPIA]`: Spanish copying, calques, and unnecessary loans.

This is the practical equivalent of mathematical reasoning primitives. The model is not rewarded for a long hidden chain of thought; it is rewarded for exposing one or two short, parseable checks that correspond to translation failure modes we can score. `translation_thinking_score()` gives extra reward for valid primitive tags while still checking bounded length and translation-specific content. Diagnostics now report average thinking primitive count, so future canaries can distinguish "formatted but empty" thinking from actual primitive use.

`scripts/train_meta_verifier_chanka_unsloth.py` trains the meta-verifier LoRA. It can either consume the generated `translation_meta_verifier_cold_start.jsonl` or build the same rows directly from the clean corpus. It saves `final_meta_verifier_lora`.

For the thinking variant, the better cold-start path is now explicit: build `self_verifiable_thinking_generator_sft.jsonl`, SFT it with `scripts/train_jsonl_sft_unsloth.py --prompt-self-verification-thinking --target-field target`, then run `self_verifiable_thinking_translation_2511` GSPO. The wrapper `experiments/gspo/run_2511_train_thinking_generator_then_gspo.sh` performs that chain. This teaches the response structure before RL, which is closer to DeepSeekMath than asking a normal translator adapter to discover the reasoning format during GSPO.

`experiments/gspo/queue_2511_thinking_generator_sft_after_active.sh` queues that branch behind active mining/training/evaluation jobs with canary-sized defaults. Use it when the GPU is occupied by meta-verifier mining but the next desired experiment is the SFT-seeded reasoning-format path. The runner now prefers meta-verifier v3 if its adapter exists and falls back to v2 only if v3 is absent; the queued GSPO canary uses 32 eval rows and evaluates at the final step to keep primitive-thinking iteration from being dominated by slow verifier evaluation.

`experiments/gspo/run_2511_self_verifiable_translation.sh` launches a small GSPO canary from the current best standalone 4B full-SFT checkpoint with a fresh LoRA adapter.

`experiments/gspo/run_2511_train_meta_verifier_then_self_gspo.sh` chains the first complete DeepSeekMath-style loop:

1. Build cold-start verifier/meta-verifier/generator data.
2. Train a meta-verifier.
3. Run self-verifiable GSPO with `--meta-verifier-adapter-path`.

`scripts/evaluate_gspo_checkpoint.py --self-verification-output` can now prompt a model for structured self-verifying translations, score only the parsed final translation, and write both `prediction` and `raw_prediction` plus the parsed `self_verification` object to JSONL. Use `--self-verification-thinking-output` to mine the bounded `Analisis de traduccion` variant; this implies self-verification parsing and keeps metrics on `Traduccion final`.

Both `scripts/evaluate_gspo_checkpoint.py` and final GSPO metrics now include self-verification diagnostics when structured output is enabled: format adherence, required-format adherence, missing-score rate, average self-score, hidden-reference true score, score gap, false-confidence rate, underconfidence rate, and bounded-thinking length/quality summaries. These diagnostics are required for the DeepSeekMath loop because reward alone can rise while the model remains badly calibrated.

Final GSPO held-out generation is batched through `generate_predictions(..., batch_size=...)` using `--per-device-eval-batch-size`. This matters for the self-verification experiments because row-by-row final decoding was a large source of queue latency after the trainer finished.

The self-verifiable launchers accept `MAX_TRAIN_SAMPLES` and `MAX_EVAL_SAMPLES`. The queue scripts default GSPO canaries to `256` train rows and `64` eval rows. Use this smaller validation surface for format/calibration iteration, then remove the caps only once the model reliably emits useful self-verification.

Structured-output parsing strips trailing `Autoevaluacion` and `Puntaje` even when the model omits the `Traduccion final:` marker. This is necessary because early self-verification adapters often emitted partial structure such as `Quechua sentence Autoevaluacion: ... Puntaje: ...`; metrics and verifier mining should score only the Quechua sentence.

## Meta-Verifier V2 Result

The first real-output meta-verifier v2 run did not improve translator quality. The original process was stopped after it had already saved `final_gspo_lora` because it was stuck in the pre-batched final metric pass. The saved adapter was evaluated with the updated batched evaluator:

```text
outputs/gspo_paper_profiles/2511_self_verifiable_translation_meta_v2_20260523-meta-v2-self/chanka_gspo/final_gspo_lora
```

Held-out metrics:

- chrF++ `38.8286`
- BLEU `5.7896`
- TER `138.2488`
- token F1 `23.2703`
- self-verification format rate `46.20%`
- missing self-score rate `53.80%`
- average self-score `1.0`
- average true score `0.2900`
- average self/true score gap `0.7687`
- false-confidence rate `97.26%`

Interpretation: the model mostly either skipped the required format or claimed perfect translations for weak outputs. This is not a usable self-verifying translator. The bounded-thinking profile plus meta-verifier-v3 mining is the next live canary because it attacks the real failure mode: absent/empty analysis and uncalibrated confidence.

`scripts/build_meta_verifier_from_self_outputs.py` converts those real model outputs into meta-verifier training rows. It compares the model's boxed self-score against the hidden-reference translation score, labels false confidence, underconfidence/hallucinated issues, missing scores, and well-calibrated analyses, and writes JSONL usable by `scripts/train_meta_verifier_chanka_unsloth.py`.

`experiments/gspo/queue_meta_verifier_v2_from_self_outputs.sh` is the first rollout-to-meta-verifier loop:

1. Wait for a self-verifiable translator adapter.
2. Mine real self-verifying outputs with `--self-verification-output`.
3. Build real meta-verifier rows from the model's own analyses.
4. Train meta-verifier v2 on cold-start + real self-output rows.
5. Run self-verifiable GSPO with meta-verifier v2.

`experiments/gspo/queue_meta_verifier_v3_from_thinking_outputs.sh` repeats the same loop for bounded-thinking outputs:

1. Wait for a `self_verifiable_thinking_translation_2511` adapter.
2. Mine real outputs with `--self-verification-thinking-output`.
3. Train meta-verifier v3 on cold-start plus real bounded-thinking failures.
4. Run another bounded-thinking GSPO pass with the v3 meta-verifier.

## Thinking vs. Self-Verification

We should use the DeepSeekMath-V2 idea of thinking, but not as an uncontrolled open `<think>` trace during GSPO. In the failed first canary, Qwen used the entire completion budget on generic English "Thinking Process" boilerplate and never reached a usable Chanka translation. That is not the DeepSeekMath mechanism we want.

For this project, the useful reasoning signal is the explicit, rewardable `Autoevaluacion`: the model states whether the translation preserved meaning, Chanka grammar, terminology, entities, and avoided Spanish copying. This can be judged by the reward and later by a learned meta-verifier. Hidden or free-form thinking is hard to score, easy to clip, and can dominate the short translation output.

Later, if we want a true long-thinking variant, use a separate profile with a larger completion budget and a parser that requires:

```text
Analisis de traduccion: ...bounded translation analysis...
Traduccion final: ...
Autoevaluacion: ...
Puntaje: \boxed{...}
```

That variant now exists as `self_verifiable_thinking_translation_2511` and is launched by `experiments/gspo/run_2511_self_verifiable_thinking_translation.sh`. It deliberately does **not** use the tokenizer's raw `<think>` mode. Instead, it asks for a short, parseable `Analisis de traduccion` field before the final translation and rewards it only when it is bounded and translation-specific. After the first canary produced ~100-token analyses with 25% clipping, the target was tightened to 1-2 checks, at most 35 words, no step-by-step rationale, and the launcher default completion cap was reduced to 112 tokens. The next revision adds explicit primitive tags such as `[SIGNIFICADO]` and `[ANTI_COPIA]`, making the reasoning field closer to DeepSeekMath/R1-style structured work while still keeping external metrics on `Traduccion final` only.

Do not scale that variant until it proves it terminates and improves final translations. The first target is a canary against the meta-verifier-v2 run, not a replacement for the current deployable model.

The first tightened bounded-thinking canary was a useful negative result. It stopped the runaway traces but did not produce a good translator:

- run: `outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_20260523-thinking-meta-v2`
- chrF++: 35.1158
- BLEU: 12.9443
- TER: 95.3757
- token F1: 18.2616
- trainer eval reward: 0.0911
- format/thinking-format rate: 45.3125%
- missing self-score rate: 54.6875%
- average self-score / true-score: 0.9552 / 0.2577
- false-confidence rate: 93.1034%
- average thinking score: 0.1742

Conclusion: the bounded thinking format is now mechanically viable, but a normal translator adapter does not reliably discover the reasoning/checking behavior from a tiny GSPO canary. The next serious branch should use the bounded-thinking generator SFT cold start before GSPO, while the meta-verifier-v3 queue mines this failed run for real false-confidence examples.

Operational note: `queue_meta_verifier_v3_from_thinking_outputs.sh` now mines with batch size 4 by default. The first restarted mining job used batch size 1 and was unnecessarily slow; the L40S had ample headroom, and the final evaluator path already proved batched decoding works for these structured outputs.

The v3 mining run from that failed adapter produced the desired hard negatives:

- run directory: `outputs/self_verification_mining/20260523-meta-v3-thinking/`
- rows: 320 generated / 317 meta-verifier records after dedupe
- chrF++: 39.4663
- BLEU: 9.0469
- token F1: 28.3023
- TER: 105.8957
- format/thinking-format rate: 86.875%
- missing self-score rate: 13.125%
- average self-score / true-score: 0.9586 / 0.3278
- false-confidence rate: 91.7266%
- label rationales: 254 false-confidence, 40 missing-score, 22 matching-analysis, 1 underconfident
- severities: 269 critical, 26 major, 18 minor, 4 none

This confirms that the main failure is not lack of self-evaluation syntax anymore; it is calibration. The meta-verifier should be trained heavily on these real false-confidence traces, then the next GSPO run should reward agreement with that meta-verifier rather than trusting the model's boxed score.

The first GSPO run using the v3 meta-verifier still failed as a translator:

- run directory: `outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_meta_v3_20260523-meta-v3-thinking`
- chrF++: 32.5425
- BLEU: 8.5679
- TER: 119.6532
- token F1: 17.9425
- trainer eval reward: 0.0970
- format/thinking-format rate: 50.0%
- missing self-score rate: 50.0%
- average self-score / true-score: 0.9250 / 0.2457
- false-confidence rate: 93.75%
- average thinking score: 0.2036

The reward rose, but corpus translation metrics collapsed. That makes this a useful negative: the model learned enough structure to satisfy part of the verifier loop, but the loop is not yet grounded strongly enough in translation quality. The next variant should give the generator a supervised head-start on the primitive-tag thinking format before GSPO.

Speed note: `scripts/train_gspo_chanka_unsloth.py` now supports `--no-trainer-eval`, `--final-metrics-max-samples`, and `--final-generation-batch-size`. For exploratory structured-output canaries, disable trainer eval and score only 16-32 final rows; reserve full trainer eval/full final metrics for runs that already show format adherence and non-collapsed translation quality.

## Frontier Thinking Data

The strongest cold-start path is synthetic primitive thinking from a stronger reasoning model, but the synthetic part should be the audit trace, not unreviewed Chanka labels. `scripts/build_frontier_thinking_sft_jsonl.py` implements this:

- defaults to `deepseek-v4-pro` through `https://api.deepseek.com`;
- reads the API key only from `DEEPSEEK_API_KEY`;
- asks for short JSON outputs with 2-4 primitive tags;
- includes reviewed few-shot primitive-check examples by default (`--few-shot-count 2`) to demonstrate the compact JSON/tag style without replacing reviewed translations;
- writes `source`, `reference`, and `target` rows compatible with `scripts/train_jsonl_sft_unsloth.py --target-field target`;
- keeps the reviewed reference as `Traduccion final` unless `--allow-model-translation` is explicitly set.

This is the safer DeepSeekMath analogue: use a frontier model to teach the student what to check (`[SIGNIFICADO]`, `[GRAMATICA]`, `[ENTIDADES]`, `[TERMINOLOGIA]`, `[ANTI_COPIA]`) while keeping the final translation anchored to our clean bilingual data. The chain script is:

```bash
cd /root/rosettia-chanka
export DEEPSEEK_API_KEY=...
FRONTIER_MAX_ROWS=128 experiments/gspo/run_deepseek_v4_pro_thinking_sft_then_gspo.sh
```

Use small batches first. Generated frontier JSONL should remain an artifact, not a committed repo file.

The frontier builder now supports a second-pass audit:

```bash
python scripts/build_frontier_thinking_sft_jsonl.py \
  --output-jsonl outputs/frontier_thinking_data/deepseek_v4_pro_thinking_sft.jsonl \
  --model deepseek-v4-pro \
  --audit \
  --audit-model deepseek-v4-flash \
  --max-rows 128
```

The audit pass returns `pass`, `score`, and `reason`, and rejects rows below `--audit-min-score`. This matters because the deterministic seed showed that primitive tags alone are not enough: the trace must make reliable, translation-grounded checks.

The main DeepSeek V4 Pro chain now runs `scripts/preflight_deepseekmath_language_loop.py` before the first API call. It writes `${DATA_DIR}/preflight_report.json` and checks API key presence, base adapter existence, required scripts, and free disk. The report records only `api_key_set`, never the secret value. Set `RUN_PREFLIGHT=false` only for debugging the shell wrapper itself.

The builder writes incrementally now: accepted rows are appended directly to the output JSONL, rejected/error rows go to a failures JSONL, and `--resume` is enabled by default. This is important for API work because a timeout halfway through should not waste already-paid frontier calls. Use `--retry-failures` only after changing the prompt/model or when the failures were transient API errors.

Before SFT, `scripts/check_frontier_thinking_data.py` now gates on accepted-row count, accept rate, and primitive coverage. The chain defaults to `MIN_FRONTIER_ROWS_FOR_SFT=64`, `MIN_FRONTIER_ACCEPT_RATE=0.50`, `MIN_FRONTIER_PRIMITIVE_TAGS_PER_ROW=2`, `MIN_FRONTIER_PRIMITIVE_ROW_RATE=0.90`, and `MIN_FRONTIER_DISTINCT_PRIMITIVES=4`; lower those only for smoke tests. This prevents training the student on a tiny, mostly rejected, or one-note frontier batch, which would make the next RL result uninterpretable.

After that gate, `scripts/report_frontier_thinking_data.py` writes a JSON and Markdown report with primitive coverage, audit scores/reasons, failure reasons, and examples from accepted and rejected rows. Inspect `deepseek_v4_pro_thinking_report.md` before interpreting the SFT run; the report is the quickest way to catch a prompt that passed row-count checks but produced unhelpful analysis traces.

The chained experiment now evaluates SFT-only before RL. It runs `scripts/evaluate_gspo_checkpoint.py --self-verification-thinking-output` after the frontier SFT and only enters GSPO if `MIN_SFT_CHRF_FOR_GSPO` and `MIN_SFT_FORMAT_FOR_GSPO` are met. If the SFT head-start already collapses translation quality or format adherence, the script exits without spending time on another GSPO collapse.

The same SFT-only evaluation is mined into meta-verifier hard cases before the GSPO gate. The chain writes `meta_hardcases_from_sft_eval.jsonl` and `meta_hardcases_from_sft_eval.summary.json` with `scripts/build_meta_verifier_from_self_outputs.py`. This keeps failed frontier SFT attempts useful: overconfident or malformed student outputs become real meta-verifier negatives for the next verifier iteration instead of just dead-end metrics.

The chain can now close that loop inside the same run. With `TRAIN_SFT_META_VERIFIER=true` it checks `MIN_SFT_META_RECORDS_FOR_TRAIN`, builds cold-start meta rows with `scripts/build_self_verifiable_translation_data.py`, trains `scripts/train_meta_verifier_chanka_unsloth.py` on cold-start plus real SFT hard cases, and then uses the refreshed meta-verifier for GSPO. That makes the ordering:

1. frontier primitive-thinking SFT data;
2. SFT head-start on the translator;
3. SFT-only self-verification evaluation;
4. real hardcase mining for false confidence and malformed scores;
5. meta-verifier refresh;
6. gated GSPO with the refreshed verifier signal.

This is still not proof that RL improves translation. It is the first version of the actual DeepSeekMath-style feedback loop for language, because the verifier is updated from student self-analysis failures instead of remaining a static synthetic judge.

After GSPO, the loop now writes the RL model's final structured predictions and mines them too. `scripts/train_gspo_chanka_unsloth.py --predictions-jsonl` emits source/reference/prediction/raw/self-verification rows, `run_2511_self_verifiable_thinking_translation.sh` writes `final_predictions.jsonl`, and the DeepSeek V4 Pro chain turns that into `meta_hardcases_from_gspo_eval.jsonl` when `MINE_GSPO_META=true`. Those post-GSPO hard cases should be treated as the seed for the next iteration's meta-verifier, because they reflect the failure modes created by the actual RL policy rather than only the SFT head-start.

The next-iteration runner is:

```bash
cd /root/rosettia-chanka
SFT_META_JSONL=outputs/deepseek_v4_pro_thinking_sft_<stamp>/sft_only_eval/meta_hardcases_from_sft_eval.jsonl \
GSPO_META_JSONL=outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_deepseek_seeded_<stamp>/chanka_gspo/meta_hardcases_from_gspo_eval.jsonl \
experiments/gspo/run_next_meta_verifier_from_hardcases.sh
```

It uses `scripts/check_meta_hardcase_data.py` to require enough valid real hardcases, adds cold-start meta rows, and trains a new meta-verifier. Feed the printed `final_meta_verifier_lora` back as `META_VERIFIER_ADAPTER` for the next GSPO cycle. This is the practical iteration boundary: data generation and SFT may be reused, while the verifier keeps absorbing the student's current failure distribution.

Then run the next policy update without regenerating frontier data:

```bash
BASE_MODEL=outputs/deepseek_v4_pro_thinking_sft_<stamp>/final_lora \
META_VERIFIER_ADAPTER=outputs/chanka_translation_meta_verifier_iter_<stamp>/final_meta_verifier_lora \
experiments/gspo/run_followup_gspo_with_meta_verifier.sh
```

That follow-up runner performs a small self-verifiable thinking GSPO pass, writes final predictions, and mines `meta_hardcases_from_followup_gspo_eval.jsonl`. If those hardcases are valid and numerous enough, feed them back into `run_next_meta_verifier_from_hardcases.sh` as `GSPO_META_JSONL` or `EXTRA_META_JSONLS`. This gives the project a concrete loop:

1. frontier thinking SFT;
2. GSPO with current meta-verifier;
3. mine SFT/GSPO self-analysis failures;
4. refresh meta-verifier;
5. GSPO again from the same thinking-SFT base or from the previous policy adapter;
6. repeat only while corpus metrics and calibration improve.

Promotion is now explicit. `scripts/check_policy_iteration_metrics.py` writes a pass/fail JSON report using chrF++/BLEU/token-F1/TER, self-verification required-format rate, false-confidence, and missing-score thresholds. `run_followup_gspo_with_meta_verifier.sh` writes this as `promotion_gate.json`; pass `BASELINE_METRICS_JSON` to compare against the previous SFT or policy metrics. For exploratory loops, leave `REQUIRE_PROMOTION=false` so the run still mines hardcases after a failed gate. For serious model promotion, set `REQUIRE_PROMOTION=true`. This guard exists because earlier runs showed trainer reward increasing while chrF++/BLEU collapsed.

The first DeepSeek V4 Pro chain now follows the same evidence protocol after its initial GSPO pass. It writes `promotion_gate.json`, compares against the SFT-only eval metrics, mines GSPO hardcases, and writes `cycle_manifest.json`. `REQUIRE_GSPO_PROMOTION=false` by default, so a failed first policy still produces useful hardcase data and an audit manifest.

The two-step iteration can also be launched as one command:

```bash
BASE_MODEL=outputs/deepseek_v4_pro_thinking_sft_<stamp>/final_lora \
SFT_META_JSONL=outputs/deepseek_v4_pro_thinking_sft_<stamp>/sft_only_eval/meta_hardcases_from_sft_eval.jsonl \
GSPO_META_JSONL=outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_deepseek_seeded_<stamp>/chanka_gspo/meta_hardcases_from_gspo_eval.jsonl \
BASELINE_METRICS_JSON=outputs/deepseek_v4_pro_thinking_sft_<stamp>/sft_only_eval/metrics.json \
experiments/gspo/run_hardcase_meta_then_followup_gspo_cycle.sh
```

This wrapper trains the next meta-verifier and immediately runs the follow-up GSPO pass with promotion gating and hardcase mining. Use it after the first DeepSeek V4 Pro frontier SFT run has produced real SFT/GSPO hardcases.

The wrapper also writes `cycle_manifest.json` in the follow-up output directory. That manifest records the base policy, refreshed meta-verifier adapter, promotion gate, final metrics, input hardcase counts, and newly mined hardcase counts. Treat this manifest as the audit record for the iteration; shell logs are secondary.

To compare several iterations:

```bash
python scripts/summarize_deepseekmath_cycles.py outputs/gspo_paper_profiles
```

This writes `cycle_summary.jsonl` and `cycle_summary.md`, ranking promoted cycles before failed cycles and including calibration plus hardcase counts. Failed cycles should be read as data-mining events unless the promotion gate passes.

## SFT-Seeded GSPO Negative Result

The deterministic primitive-thinking SFT seed did not fix the RL collapse:

- run directory: `outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_sft_seeded_20260523-thinking-generator-seeded`
- source seed: `outputs/self_verifiable_thinking_generator_sft_20260523-thinking-generator-seeded/final_lora`
- final metric rows: 16 of 32 held-out rows
- chrF++: 18.5127
- BLEU: 0.6980
- TER: 302.5641
- token F1: 6.1756
- format/thinking-format rate: 18.75%
- missing self-score rate: 81.25%
- false-confidence rate: 100.0%
- average thinking primitive count: 2.125

This is worse than the non-seeded v3 run. The lesson is that deterministic template SFT teaches the model to mention the primitive tags, but not to use them as reliable translation checks. Before more RL, evaluate SFT-only outputs and replace the deterministic traces with frontier-generated, audited, concise translation checks. The frontier traces should still keep the reviewed reference as the final translation unless a human reviews model-proposed alternatives.

## Why This Fits Chanka Better Than Plain BLEU RL

BLEU/chrF rewards are useful but too shallow for the user's goal of learning grammar structures. A translation can gain n-gram overlap while still copying Spanish structure, omitting agglutinative morphology, or choosing a misleading Chanka term. The DeepSeekMath-V2 adaptation makes the model expose its own quality judgement, then rewards calibration. This creates pressure toward internal checks like:

- Did I preserve named entities and numbers?
- Did I leave Spanish stopwords or source word order?
- Did I omit a predicate, negation, possessive, evidential, or case relation?
- Did I invent unsupported content?
- Is this a natural Chanka sentence rather than a bilingual word salad?

This still does not magically prove grammatical understanding. It gives us an RL loop where the model is rewarded for finding and fixing translation-specific issues, which is closer to the "grammar structures like R1/math structures" idea than optimizing a scalar metric alone.

## Important Limitation

The current `self_verifiable_translation_2511` reward can use a trained meta-verifier via `--meta-verifier-adapter-path`, but the first available meta-verifier data is synthetic/cold-start. That is closer to DeepSeekMath-V2 than a deterministic proxy, but not yet the full loop. The paper's strongest loop repeatedly mines hard generator/verifier failures and retrains the verifier/meta-verifier.

Next serious step:

1. Build cold-start data with `scripts/build_self_verifiable_translation_data.py`.
2. Train a translation verifier on candidate/reference/source triples.
3. Train a meta-verifier on verifier analyses.
4. Replace or blend the deterministic `self_analysis_meta_score` with the learned meta-verifier score.
5. Mine hard examples from K-sample outputs of the current best translator, especially cases with high self-score but low hidden-reference score. `queue_meta_verifier_v2_from_self_outputs.sh` implements the first version.
6. Iterate verifier -> generator GSPO -> hard-case mining -> verifier/meta-verifier.

## Canary Command

On the remote training host:

```bash
cd /root/rosettia-chanka
experiments/gspo/run_2511_self_verifiable_translation.sh
```

Useful overrides:

```bash
MAX_STEPS=64 EVAL_STEPS=16 SAVE_STEPS=16 \
LEARNING_RATE=2e-7 NUM_GENERATIONS=4 \
experiments/gspo/run_2511_self_verifiable_translation.sh
```

Keep this as a canary until we confirm the model reliably follows the structured format. If it spends too many tokens on self-analysis, reduce `MAX_COMPLETION_LENGTH` or shorten the prompt. If `Thinking Process:` or open `<think>` tags reappear, check that `force_tokenizer_no_thinking_template()` printed its patch message before training.

To train and use the cold-start meta-verifier:

```bash
cd /root/rosettia-chanka
experiments/gspo/run_2511_train_meta_verifier_then_self_gspo.sh
```

To mine real self-analysis failures and train meta-verifier v2:

```bash
cd /root/rosettia-chanka
experiments/gspo/queue_meta_verifier_v2_from_self_outputs.sh
```

To test the bounded-thinking variant:

```bash
cd /root/rosettia-chanka
META_VERIFIER_ADAPTER=outputs/chanka_translation_meta_verifier_v2_20260523-meta-v2-self/final_meta_verifier_lora \
MAX_STEPS=8 EVAL_STEPS=4 SAVE_STEPS=4 \
experiments/gspo/run_2511_self_verifiable_thinking_translation.sh
```

If another GSPO process is still using the GPU, queue it with:

```bash
cd /root/rosettia-chanka
experiments/gspo/queue_2511_self_verifiable_thinking_after_current_gspo.sh
```

To mine bounded-thinking failures and close the next loop:

```bash
cd /root/rosettia-chanka
experiments/gspo/queue_meta_verifier_v3_from_thinking_outputs.sh
```

## Frontier Primitive Requirements

The DeepSeek V4 Pro synthetic-thinking builder now makes primitive tags row-specific instead of only asking for "2 to 4 tags" globally. For each reviewed source/reference pair, `scripts/build_frontier_thinking_sft_jsonl.py` derives expected tags:

- Always `[SIGNIFICADO]` and `[GRAMATICA]`, because every row should teach meaning preservation and Chanka-form checks.
- Add `[ENTIDADES]` when the row has names, numbers, or similar entity signals.
- Add `[TERMINOLOGIA]` for legal/domain terms when there is no stronger entity signal.
- Otherwise add `[ANTI_COPIA]` to keep the model alert to Spanish-copy failure modes.

The builder passes those expected tags in the DeepSeek prompt and rejects rows that omit them by default. Use `--no-require-expected-primitives` only for debugging or prompt exploration, not for serious SFT data. The pre-SFT gate also checks accepted JSONLs with `--min-expected-primitive-coverage`; `experiments/gspo/run_deepseek_v4_pro_thinking_sft_then_gspo.sh` defaults this to `MIN_FRONTIER_EXPECTED_PRIMITIVE_COVERAGE=0.95`.

The same builder also stratifies the selected source rows by expected primitives before any API call. After the seeded shuffle, `--stratify-primitives` greedily chooses rows that cover currently underrepresented tags, which gives small DeepSeek batches a better chance of including entity, terminology, grammar, meaning, and anti-copy cases. The chain leaves this on by default and exposes `FRONTIER_STRATIFY_PRIMITIVES=false` only for exact random-slice debugging.

The chain writes a pre-API source-selection report before preflight and before any DeepSeek request:

```text
${DATA_DIR}/deepseek_v4_pro_source_selection_report.json
${DATA_DIR}/deepseek_v4_pro_source_selection_report.md
```

This is produced through the builder's `--selection-only` mode, so it still works when `DEEPSEEK_API_KEY` is absent. Use it to inspect selected Spanish/Chanka rows and expected primitive counts before approving a paid frontier generation run.

The selection report is now gated by `scripts/check_frontier_selection_report.py` before preflight or API calls. The chain writes:

```text
${DATA_DIR}/deepseek_v4_pro_source_selection_gate.json
```

Default gate values require enough rows for SFT, all five expected primitive families, average expected primitives per row >= `2.5`, and at least one selected row for every primitive tag. Disable with `RUN_FRONTIER_SELECTION_GATE=false` only when debugging the wrapper itself.

The builder also dedupes exact source/reference pairs after the seeded shuffle and before primitive-balanced row selection. The selection report records `input_rows` and `duplicate_input_rows`, so repeated augmented examples do not waste DeepSeek calls or make the primitive curriculum look more diverse than it is.

The selection report is resume-aware too. It records already accepted rows, already failed rows, pending rows, and estimated frontier requests. With audit enabled, `estimated_max_frontier_requests` is at most `pending_rows * 2`, because each accepted generation can trigger a second audit call. The selection gate can enforce a hard cap through `--max-estimated-frontier-requests`; the chain exposes this as `MAX_FRONTIER_SELECTION_REQUESTS` and leaves it disabled by default (`0`). Use it for small paid smokes or when resuming partial frontier batches.

This is a small but important step toward the actual "DeepSeekMath for language" target: the synthetic trace should teach reusable translation primitives, not just a fixed response shape. If the first frontier batch fails this gate, inspect `deepseek_v4_pro_thinking_report.md` for missing expected tags before spending GPU time.

For a faster smoke:

```bash
MAX_META_SOURCE_ROWS=32 META_MAX_STEPS=4 META_EVAL_STEPS=2 META_SAVE_STEPS=2 \
GSPO_MAX_STEPS=2 GSPO_EVAL_STEPS=1 GSPO_SAVE_STEPS=1 \
experiments/gspo/run_2511_train_meta_verifier_then_self_gspo.sh
```
