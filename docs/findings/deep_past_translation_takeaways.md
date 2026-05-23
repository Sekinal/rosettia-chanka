# Deep Past Translation Takeaways

Source: user-provided Kaggle Deep Past Challenge solution writeups, reviewed on 2026-05-22.

## Actionable Ideas

The biggest transferable point is that the winning systems were data systems first and model systems second. For RosettIA-Chanka, the most useful ideas are:

- Build targeted synthetic drills from our Chanka glossary instead of generic sentence generation. Useful drill families: vocabulary-in-context, ambiguity contrasts, short formulaic legal/interview templates, and negative examples where a Spanish-looking Quechua output must be rejected.
- Use slot-fill templates for stable domains already visible in our data: identity questions, family/marital status, court procedure, location, age, dates, and simple legal formulas. These can scale cleanly if slots are curated from real Chanka references.
- Keep MBR-style candidate selection in the inference stack. The Kaggle winners used MBR or reward-model selection; our own K16 experiments show the candidate pool contains much better translations than greedy selection.
- Prefer iterative data repair over blind scaling. The most relevant loop is: generate/evaluate candidate translations, identify rows with high candidate disagreement or severe reference mismatch, then manually/LLM-review those rows and add corrected examples.
- Use pseudo-labels only when confidence is strong. Our earlier K16 MBR self-training agrees with the writeups: unfiltered pseudo-labels are risky, while conservative MBR+clean-anchor mixtures can help.
- Treat checkpoint selection as a real variable. Do not trust `final_lora`; repeatedly our best checkpoints were early. Eval loss, held-out generation, and qualitative samples all matter.
- Try weight averaging only after we have several nearby strong LoRA checkpoints or multiple small seed variants. This is now queued for the 9B Chanka path via `experiments/sft/queue_qwen35_9b_checkpoint_soup_eval.sh`, which waits for the main 9B rerank work and then evaluates a top-three weighted LoRA soup.

## Less Directly Useful

- ByT5/mT5 full fine-tuning was central in the Kaggle competition, but our current stack is Unsloth decoder-only LoRA on Qwen3.5. The principle transfers; the exact architecture choice does not.
- CPT with long warmup/stable LR is only attractive if we assemble a much larger Chanka/Ayacucho monolingual or parallel corpus. With the current clean corpus size, CPT-style runs are likely to memorize or drift.
- Backtranslation is plausible only after we have a stronger reverse Chanka->Spanish model or a verifier that can reject bad pseudo-Chanka. Otherwise it will amplify wrong morphology and Spanish leakage.

## Current Priority From These Notes

Near-term priority should be:

1. Deployable reranking from K16 candidates.
2. Glossary-triggered contextual synthetic drills, not standalone term-pair SFT.
3. More reviewed Chanka sources or reviewed extraction from official bilingual PDFs.
4. Preference/listwise training only after the selector labels are better than naive oracle-pair DPO.
