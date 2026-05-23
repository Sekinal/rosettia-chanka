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

`experiments/gspo/run_2511_self_verifiable_translation.sh` launches a small GSPO canary from the current best standalone 4B full-SFT checkpoint with a fresh LoRA adapter.

## Thinking vs. Self-Verification

We should use the DeepSeekMath-V2 idea of thinking, but not as an uncontrolled open `<think>` trace during GSPO. In the failed first canary, Qwen used the entire completion budget on generic English "Thinking Process" boilerplate and never reached a usable Chanka translation. That is not the DeepSeekMath mechanism we want.

For this project, the useful reasoning signal is the explicit, rewardable `Autoevaluacion`: the model states whether the translation preserved meaning, Chanka grammar, terminology, entities, and avoided Spanish copying. This can be judged by the reward and later by a learned meta-verifier. Hidden or free-form thinking is hard to score, easy to clip, and can dominate the short translation output.

Later, if we want a true long-thinking variant, use a separate profile with a larger completion budget and a parser that requires:

```text
<think>...bounded translation analysis...</think>
Traduccion final: ...
Autoevaluacion: ...
Puntaje: \boxed{...}
```

Do not scale that variant until it proves it terminates and improves final translations. The current canary intentionally trains structured self-verification first.

## Why This Fits Chanka Better Than Plain BLEU RL

BLEU/chrF rewards are useful but too shallow for the user's goal of learning grammar structures. A translation can gain n-gram overlap while still copying Spanish structure, omitting agglutinative morphology, or choosing a misleading Chanka term. The DeepSeekMath-V2 adaptation makes the model expose its own quality judgement, then rewards calibration. This creates pressure toward internal checks like:

- Did I preserve named entities and numbers?
- Did I leave Spanish stopwords or source word order?
- Did I omit a predicate, negation, possessive, evidential, or case relation?
- Did I invent unsupported content?
- Is this a natural Chanka sentence rather than a bilingual word salad?

This still does not magically prove grammatical understanding. It gives us an RL loop where the model is rewarded for finding and fixing translation-specific issues, which is closer to the "grammar structures like R1/math structures" idea than optimizing a scalar metric alone.

## Important Limitation

The current `self_verifiable_translation_2511` reward uses a deterministic meta-analysis proxy. It is not yet the full DeepSeekMath-V2 setup, because the paper's strongest loop uses a trained meta-verifier to judge verifier/self-analysis faithfulness.

Next serious step:

1. Build cold-start data with `scripts/build_self_verifiable_translation_data.py`.
2. Train a translation verifier on candidate/reference/source triples.
3. Train a meta-verifier on verifier analyses.
4. Replace or blend the deterministic `self_analysis_meta_score` with the learned meta-verifier score.
5. Mine hard examples from K-sample outputs of the current best translator, especially cases with high self-score but low hidden-reference score.
6. Iterate verifier -> generator GSPO -> hard-case mining -> verifier.

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
