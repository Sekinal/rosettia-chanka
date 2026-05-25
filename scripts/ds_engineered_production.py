"""DS Flash production: engineered reasoning traces for the 897 train Chanka sources.

Leak protection:
- Loads Chanka data and applies the SAME split as evaluate_gspo_checkpoint.py
  (validation_fraction=0.15, seed=3407).
- Generates ONLY for the 897 train sources. Eval sources never touched.
- Few-shot exemplars are also drawn from train rows.

High concurrency (20 workers) + retry with backoff.
"""
from __future__ import annotations
import os, sys, json, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request, error

sys.path.insert(0, "/root/rosettia-chanka")
from scripts import train_gspo_chanka_unsloth as gspo

DS_KEY = os.environ["DEEPSEEK_API_KEY"]
DS_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"
CONCURRENCY = 20
MAX_TOKENS = 2500
TEMPERATURE = 0.3
MAX_RETRIES = 3
OUTPUT_JSONL = "/root/rosettia-chanka/outputs/engineered_reasoning_data_20260525/engineered_reasoning_train.jsonl"
REJECTED_JSONL = "/root/rosettia-chanka/outputs/engineered_reasoning_data_20260525/rejected.jsonl"
RAW_JSONL = "/root/rosettia-chanka/outputs/engineered_reasoning_data_20260525/raw_responses.jsonl"
SUMMARY_JSON = "/root/rosettia-chanka/outputs/engineered_reasoning_data_20260525/summary.json"

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

# === LOAD WITH SAME SPLIT AS EVAL ===
rows = gspo.load_chanka_rows(gspo.DATASET_REPO, gspo.CHANKA_FILE)
train_rows, eval_rows = gspo.split_rows(rows, validation_fraction=0.15, seed=3407,
                                         max_train_samples=None, max_eval_samples=None)
eval_sources = {r["source"] for r in eval_rows}
print(f"loaded total={len(rows)} train={len(train_rows)} eval={len(eval_rows)}")
print(f"eval sources locked: {len(eval_sources)}")

# Some Spanish sources appear in both train and eval (parquet has duplicates
# from slash-alternative splitting). Filter train_rows to exclude any source
# that also appears in eval, so no eval source is ever sent to the API or
# used as training data downstream.
before = len(train_rows)
train_rows = [r for r in train_rows if r["source"] not in eval_sources]
removed = before - len(train_rows)
print(f"LEAK FILTER: removed {removed} train rows whose source also appears in eval; train kept={len(train_rows)}")
# Sanity check
for r in train_rows:
    assert r["source"] not in eval_sources
print("LEAK CHECK PASSED")

# Few-shot from train only
fewshot = []
for r in train_rows:
    src = r["source"]
    tgt = r["target"]
    if 3 <= len(src.split()) <= 6 and len(tgt.split()) <= 6 and src not in eval_sources:
        fewshot.append((src, tgt))
        if len(fewshot) == 3:
            break

SYSTEM = (
    "Eres un lingüista experto en gramática del quechua chanka (variedad quy del Quechua "
    "sureño hablada en Ayacucho, Apurímac y Huancavelica). Tu tarea es crear trazas de "
    "razonamiento paso a paso que enseñen cómo se construye una traducción del español "
    "al quechua chanka. Debes explicar la morfología real (raíces, sufijos de caso, "
    "persona, tiempo, evidencial) y no inventar reglas. La traducción final debe coincidir "
    "exactamente con la traducción esperada que se te proporciona."
)


def format_user_prompt(src: str, gold: str) -> str:
    fewshot_block = ""
    for i, (es, qu) in enumerate(fewshot, 1):
        fewshot_block += (
            f"\nEjemplo {i}\n"
            f"Español: {es}\n"
            f"Traducción esperada en chanka: {qu}\n"
            f"Paso 1 - Análisis del español: <breve>\n"
            f"Paso 2 - Glosa palabra por palabra hacia chanka: <raíz + sufijos>\n"
            f"Paso 3 - Composición literal: <morfemas concatenados>\n"
            f"Paso 4 - Reglas morfológicas chanka aplicadas: <reglas reales>\n"
            f"Final: {qu}\n"
            f"Autoevaluación: <una o dos oraciones>\n"
            f"Puntaje: \\boxed{{0.95}}\n"
        )
    return (
        "Para cada par fuente-objetivo, produce una traza de razonamiento usando "
        "exactamente este formato de 4 pasos seguidos de Final, Autoevaluación y Puntaje. "
        "La línea 'Final:' debe terminar con la traducción esperada literalmente. "
        "No agregues texto antes de 'Paso 1' ni después de la línea de Puntaje."
        + fewshot_block
        + "\n\nAhora produce SOLO la traza (sin repetir las cabeceras 'Español:' o "
        f"'Traducción esperada en chanka:') para:\n"
        f"Español: {src}\n"
        f"Traducción esperada en chanka: {gold}\n"
    )


def call_ds_flash(src: str, gold: str, timeout: int = 120) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": format_user_prompt(src, gold)},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    data = json.dumps(payload).encode()
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            req = request.Request(
                DS_URL,
                data=data,
                headers={"Authorization": "Bearer " + DS_KEY, "Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=timeout) as r:
                resp = json.load(r)
            text = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
            finish = resp["choices"][0].get("finish_reason", "")
            return {"source": src, "gold": gold, "trace": text, "usage": usage, "finish_reason": finish, "ok": True}
        except error.HTTPError as e:
            body = e.read()[:300].decode(errors="replace")
            last_err = f"HTTPError {e.code}: {body}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt + random.random())
                continue
            return {"source": src, "gold": gold, "ok": False, "error": last_err}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt + random.random())
    return {"source": src, "gold": gold, "ok": False, "error": last_err}


# Normalize a string for the gold-match check.
def normalize(s: str) -> str:
    return " ".join(s.split())


def extract_final(trace: str) -> str | None:
    import re
    # Find last line that starts with "Final:" or "Final " (case-insensitive)
    matches = list(re.finditer(r"(?im)^\s*final\s*:\s*(.+?)\s*$", trace))
    if not matches:
        return None
    return matches[-1].group(1).strip()


# Run production
all_train = [r for r in train_rows if r["source"] not in eval_sources]
# Skip few-shot exemplars (already used in prompt)
fewshot_pairs = set((s, t) for s, t in fewshot)
targets = [r for r in all_train if (r["source"], r["target"]) not in fewshot_pairs]
print(f"\nTargets for DS Flash: {len(targets)}")

t0 = time.time()
ok_rows = []
rej_rows = []
raw_records = []
with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    futures = {pool.submit(call_ds_flash, r["source"], r["target"]): r for r in targets}
    done = 0
    for fut in as_completed(futures):
        res = fut.result()
        done += 1
        raw_records.append(res)
        if not res["ok"]:
            rej_rows.append({**res, "reject_reason": "api_error"})
        else:
            trace = res["trace"]
            final = extract_final(trace)
            if final is None:
                rej_rows.append({**res, "reject_reason": "no_final_line", "extracted_final": None})
            elif normalize(final).lower() != normalize(res["gold"]).lower():
                rej_rows.append({**res, "reject_reason": "final_mismatch", "extracted_final": final})
            elif res.get("finish_reason") == "length":
                rej_rows.append({**res, "reject_reason": "finish_reason_length", "extracted_final": final})
            else:
                ok_rows.append({**res, "extracted_final": final})
        if done % 25 == 0 or done == len(targets):
            elapsed = time.time() - t0
            rate = done / max(1e-9, elapsed)
            eta = (len(targets) - done) / max(1e-9, rate)
            print(f"  done={done}/{len(targets)} ok={len(ok_rows)} rej={len(rej_rows)} elapsed={elapsed:.0f}s eta={eta:.0f}s")

dt = time.time() - t0
print(f"\nFinished in {dt:.1f}s; ok={len(ok_rows)} rejected={len(rej_rows)}")

# Save accepted in SFT-ready JSONL
with open(OUTPUT_JSONL, "w") as f:
    for r in ok_rows:
        f.write(json.dumps({
            "source": r["source"],
            "reference": r["gold"],
            "target": r["trace"],
            "extracted_final": r["extracted_final"],
            "source_name": "clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet",
            "variant": "quy/chanka",
            "task": "engineered_reasoning_translation_generation",
            "prompt_mode": "engineered_reasoning",
            "usage": r.get("usage", {}),
        }, ensure_ascii=False) + "\n")

with open(REJECTED_JSONL, "w") as f:
    for r in rej_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(RAW_JSONL, "w") as f:
    for r in raw_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Token usage / cost estimate
total_prompt = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in raw_records if r.get("ok"))
total_completion = sum(r.get("usage", {}).get("completion_tokens", 0) for r in raw_records if r.get("ok"))
print(f"\nTokens used (successful calls): prompt={total_prompt:,} completion={total_completion:,}")

summary = {
    "model": MODEL,
    "concurrency": CONCURRENCY,
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "targets": len(targets),
    "accepted": len(ok_rows),
    "rejected": len(rej_rows),
    "reject_reasons": {},
    "total_prompt_tokens": total_prompt,
    "total_completion_tokens": total_completion,
    "elapsed_seconds": dt,
    "eval_sources_excluded": len(eval_sources),
    "fewshot_examples": [{"source": s, "target": t} for s, t in fewshot],
    "leak_check": "passed",
}
for r in rej_rows:
    reason = r.get("reject_reason", "?")
    summary["reject_reasons"][reason] = summary["reject_reasons"].get(reason, 0) + 1

with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\nSummary:", json.dumps(summary["reject_reasons"]))
print(f"Wrote {OUTPUT_JSONL}")
