"""DS Flash multi-flavor smoke: generate ~50 traces in each of 3 styles
to compare quality before scaling. Strictly leak-free.

Flavors:
  A: morphological_verbose - explicit 4-step linguistic breakdown (already running)
  B: compact_gloss - interlinear morpheme glossing, minimal prose
  C: natural_cot - conversational chain-of-thought in Spanish

Each row carries `prompt_mode=engineered_reasoning_<flavor>` so we can train
matched SFT canaries downstream.
"""
from __future__ import annotations
import os, sys, json, time, random, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request, error

sys.path.insert(0, "/root/rosettia-chanka")
from scripts import train_gspo_chanka_unsloth as gspo

DS_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"


SYSTEM = (
    "Eres un lingüista experto en gramática del quechua chanka (variedad quy del Quechua "
    "sureño hablada en Ayacucho, Apurímac y Huancavelica). Tu tarea es crear trazas de "
    "razonamiento paso a paso que enseñen cómo se construye una traducción del español "
    "al quechua chanka. Debes explicar la morfología real (raíces, sufijos de caso, "
    "persona, tiempo, evidencial) y no inventar reglas. La traducción final debe coincidir "
    "exactamente con la traducción esperada que se te proporciona."
)


def user_prompt_compact_gloss(src: str, gold: str, fewshot: list[tuple[str, str]]) -> str:
    fewshot_block = ""
    for i, (es, qu) in enumerate(fewshot, 1):
        fewshot_block += (
            f"\nEjemplo {i}\n"
            f"Español: {es}\n"
            f"Glosa morfémica:\n"
            f"- <palabra>: <raíz>[<significado>]-<sufijo>[<función>] (una línea por palabra)\n"
            f"Composición: <morfemas concatenados>\n"
            f"Final: {qu}\n"
            f"Puntaje: \\boxed{{0.95}}\n"
        )
    return (
        "Produce una glosa compacta (sin prosa larga) para enseñar al modelo. "
        "Limita la respuesta a: cabecera 'Glosa morfémica:' con una línea por palabra del "
        "español, luego 'Composición:', 'Final:' (con la traducción exacta esperada) y "
        "'Puntaje:'. Sin 'Paso 1', sin explicaciones largas. Tamaño objetivo: 100-250 tokens.\n"
        + fewshot_block
        + f"\nAhora produce SOLO la glosa para:\nEspañol: {src}\nTraducción esperada en chanka: {gold}\n"
    )


def user_prompt_natural_cot(src: str, gold: str, fewshot: list[tuple[str, str]]) -> str:
    fewshot_block = ""
    for i, (es, qu) in enumerate(fewshot, 1):
        fewshot_block += (
            f"\nEjemplo {i}\n"
            f"Español: {es}\n"
            f"Pensando: primero entiendo qué significa la frase, luego cómo se diría "
            f"palabra por palabra, después aplico las reglas del chanka...\n"
            f"<una a tres oraciones de razonamiento natural en primera persona>\n"
            f"Final: {qu}\n"
            f"Puntaje: \\boxed{{0.95}}\n"
        )
    return (
        "Produce un razonamiento natural en primera persona y en español, como un humano "
        "pensando en voz alta. Estructura informal pero con contenido morfológico real. "
        "Termina con 'Final: <traducción>' y 'Puntaje: \\boxed{...}'. Tamaño objetivo: "
        "150-350 tokens.\n"
        + fewshot_block
        + f"\nAhora pensemos sobre:\nEspañol: {src}\nTraducción esperada en chanka: {gold}\n"
    )


FLAVOR_BUILDERS = {
    "compact_gloss": user_prompt_compact_gloss,
    "natural_cot": user_prompt_natural_cot,
}


def call_ds(src: str, gold: str, user_prompt: str, max_tokens: int, key: str,
            timeout: int = 90, max_retries: int = 3) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    data = json.dumps(payload).encode()
    last_err = None
    for attempt in range(max_retries):
        try:
            req = request.Request(
                DS_URL, data=data,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=timeout) as r:
                resp = json.load(r)
            text = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
            finish = resp["choices"][0].get("finish_reason", "")
            return {"source": src, "gold": gold, "trace": text, "usage": usage,
                    "finish_reason": finish, "ok": True}
        except error.HTTPError as e:
            last_err = f"HTTPError {e.code}: {e.read()[:200].decode(errors='replace')}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt + random.random())
                continue
            return {"source": src, "gold": gold, "ok": False, "error": last_err}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt + random.random())
    return {"source": src, "gold": gold, "ok": False, "error": last_err}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", choices=list(FLAVOR_BUILDERS.keys()), required=True)
    parser.add_argument("--max-sources", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    key = os.environ["DEEPSEEK_API_KEY"]

    # Same leak-proof loader
    rows = gspo.load_chanka_rows(gspo.DATASET_REPO, gspo.CHANKA_FILE)
    train_rows, eval_rows = gspo.split_rows(rows, validation_fraction=0.15, seed=3407,
                                             max_train_samples=None, max_eval_samples=None)
    eval_sources = {r["source"] for r in eval_rows}
    train_rows = [r for r in train_rows if r["source"] not in eval_sources]
    print(f"train rows leak-free: {len(train_rows)}, eval sources excluded: {len(eval_sources)}")

    # Three few-shot exemplars from train
    fewshot = []
    for r in train_rows:
        s, t = r["source"], r["target"]
        if 3 <= len(s.split()) <= 6 and len(t.split()) <= 6:
            fewshot.append((s, t))
            if len(fewshot) == 3:
                break
    fewshot_set = {(s, t) for s, t in fewshot}

    targets = [r for r in train_rows if (r["source"], r["target"]) not in fewshot_set][:args.max_sources]
    print(f"flavor={args.flavor} targets={len(targets)} max_tokens={args.max_tokens}")

    builder = FLAVOR_BUILDERS[args.flavor]
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(call_ds, r["source"], r["target"],
                        builder(r["source"], r["target"], fewshot),
                        args.max_tokens, key): r
            for r in targets
        }
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            results.append(res)
            if done % 10 == 0:
                print(f"  done={done}/{len(targets)} ok={sum(1 for r in results if r['ok'])}")

    print(f"finished in {time.time()-t0:.1f}s")

    # Extract Final and validate
    import re
    def extract_final(trace):
        ms = list(re.finditer(r"(?im)^\s*final\s*:\s*(.+?)\s*$", trace))
        return ms[-1].group(1).strip() if ms else None

    def norm(s): return " ".join(s.split())

    ok_rows, rej_rows = [], []
    for r in results:
        if not r["ok"]:
            rej_rows.append({**r, "reject_reason": "api_error"})
            continue
        final = extract_final(r["trace"])
        if final is None:
            rej_rows.append({**r, "reject_reason": "no_final_line"})
        elif norm(final).lower() != norm(r["gold"]).lower():
            rej_rows.append({**r, "reject_reason": "final_mismatch", "extracted_final": final})
        elif r.get("finish_reason") == "length":
            rej_rows.append({**r, "reject_reason": "finish_reason_length", "extracted_final": final})
        else:
            ok_rows.append({**r, "extracted_final": final})

    with open(f"{args.output_dir}/accepted.jsonl", "w") as f:
        for r in ok_rows:
            f.write(json.dumps({
                "source": r["source"],
                "reference": r["gold"],
                "target": r["trace"],
                "extracted_final": r["extracted_final"],
                "source_name": "clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet",
                "variant": "quy/chanka",
                "task": f"engineered_reasoning_{args.flavor}",
                "prompt_mode": f"engineered_reasoning_{args.flavor}",
                "usage": r.get("usage", {}),
            }, ensure_ascii=False) + "\n")
    with open(f"{args.output_dir}/rejected.jsonl", "w") as f:
        for r in rej_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_prompt = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results if r.get("ok"))
    total_completion = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results if r.get("ok"))
    summary = {
        "flavor": args.flavor,
        "targets": len(targets),
        "accepted": len(ok_rows),
        "rejected": len(rej_rows),
        "reject_reasons": {},
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "elapsed_seconds": time.time() - t0,
        "model": MODEL,
        "max_tokens": args.max_tokens,
    }
    for r in rej_rows:
        reason = r.get("reject_reason", "?")
        summary["reject_reasons"][reason] = summary["reject_reasons"].get(reason, 0) + 1
    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("summary:", json.dumps(summary["reject_reasons"]), "accepted:", len(ok_rows))


if __name__ == "__main__":
    main()
