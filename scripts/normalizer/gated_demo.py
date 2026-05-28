import json, re, sys
sys.path.insert(0,"/root/rosettia-chanka/scripts/normalizer")
SYSTEM=("You are an expert Chanka (Ayacucho) Quechua orthographic normalizer "
"following the MINEDU 2021 standard. For each input, emit a <think>...</think> "
"trace that lists every token left-to-right with the spec rule cited (R1-R7, "
"S1-S8, L0-L3, sections 6.5, 6.6, 8.5, 8.6), then a Normalized: line with the canonical sentence.")
def extract(t):
    b=re.sub(r"<think>.*?</think>","",t,flags=re.DOTALL)
    m=re.search(r"Normalized\s*:\s*(.+?)(?:\n|$)",b,flags=re.DOTALL)
    return m.group(1).strip().strip('"') if m else b.strip()
def safe_tok(tok):
    o=tok.replace("'","").replace("’","")
    for a,b in [("chh","ch"),("kh","k"),("ph","p"),("qh","q"),("th","t")]: o=o.replace(a,b)
    o=o.replace("e","i").replace("o","u").replace("E","I").replace("O","U")
    return re.sub(r"(?<!l)l(q)",r"ll\1",o)
def gate(orig,prop):
    ot,pt=orig.split(),prop.split()
    if len(ot)!=len(pt): return orig
    return " ".join(b if (a==b or b==safe_tok(a)) else a for a,b in zip(ot,pt))
def n_corrupt(orig,prop):
    o,p=orig.split(),prop.split()
    if len(o)!=len(p): return 1
    return sum(1 for a,b in zip(o,p) if a!=b and b!=safe_tok(a))

def main():
    clean=[l.strip() for l in open("/root/rosettia-chanka/data/normalizer_precision_holdout.txt") if l.strip()][:200]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained("unsloth/Qwen3.5-4B",trust_remote_code=True)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    llm=LLM(model="unsloth/Qwen3.5-4B",enable_lora=True,max_lora_rank=512,dtype="bfloat16",
            max_model_len=4096,gpu_memory_utilization=0.85,trust_remote_code=True,enforce_eager=True)
    lora=LoRARequest("n",1,"/root/rosettia-chanka/outputs/v45a_normalizer_20260528/checkpoint-1263")
    sp=SamplingParams(temperature=0.0,max_tokens=512,stop=["<|im_end|>","<|endoftext|>"])
    prompts=[tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":f"Normalize this Chanka sentence per the MINEDU 2021 spec:\n\n{s}"}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for s in clean]
    outs=llm.generate(prompts,sp,lora_request=lora)
    rc=gc=rch=gch=0
    samples=[]
    for s,o in zip(clean,outs):
        pred=extract(o.outputs[0].text); g=gate(s,pred)
        if n_corrupt(s,pred)>0: rc+=1
        if n_corrupt(s,g)>0: gc+=1
        if pred.strip()!=s.strip(): rch+=1
        if g.strip()!=s.strip():
            gch+=1
            if len(samples)<8: samples.append((s,g))
    N=len(clean)
    print(f"\n==== ckpt-1263 on {N} clean v30-holdout ====")
    print(f"RAW model: corrupted {rc}/{N} ({rc/N*100:.1f}%), changed {rch}")
    print(f"GATED:     corrupted {gc}/{N} ({gc/N*100:.1f}%), changed {gch}")
    print("GATED sample changes (should be genuine safe transforms):")
    for s,g in samples:
        d="  ".join(f"{a}->{b}" for a,b in zip(s.split(),g.split()) if a!=b)
        print(f"   {d}")

if __name__ == "__main__":
    main()
