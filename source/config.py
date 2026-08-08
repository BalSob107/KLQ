import os

ROOT = "./"

# MODELS[key]: name of the model, refer to the model by this name in the other stages (--model {name})
# hf: HuggingFace repo id
# tag: short name used in cache filenames
# NB: number of hidden layers
# D: hidden size
# DI: intermediate size
# H_KV: n of kv heads
# HEAD_DIM: dimensions per head
# VOCAB: vocab size
# VAL_DTYPE: uint32 if VOCAB > 65535 else uint16
# VAL_BIN: token filename built by the pretokenization stage

MODELS = {
    "smollm2": dict(
        hf="HuggingFaceTB/SmolLM2-135M",
        tag="135M",
        NB=30,          # transformer blocks
        D=576, DI=1536,
        H_KV=3, HEAD_DIM=64,
        VOCAB=49152,
        VAL_DTYPE="uint16",
        VAL_BIN=os.path.join(ROOT, "data", "val_smollm2_135m.bin"),
    ),
    "qwen": dict(
        hf="Qwen/Qwen2.5-0.5B",
        tag="qwen0.5B",
        NB=24,
        D=896, DI=4864,
        H_KV=2, HEAD_DIM=64,    
        VOCAB=151936,            
        VAL_DTYPE="uint32",
        VAL_BIN=os.path.join(ROOT, "val_qwen_25_05b.bin"), 
    ),
    "llama3_1b": dict(
        hf="unsloth/Llama-3.2-1B",   
        tag="llama3_1b",
        NB=16,
        D=2048, DI=8192,
        H_KV=8, HEAD_DIM=64,   
        VOCAB=128256,
        VAL_DTYPE="uint32",
        VAL_BIN=os.path.join(ROOT, "data", "val_llama_3_1b.bin"),  
    ),
}

CACHE_DIR = os.path.join(ROOT, "klq_cache")

def paths(model):
    cfg = MODELS[model]
    d = os.path.join(CACHE_DIR, cfg["tag"])
    os.makedirs(d, exist_ok=True)
    return dict(
        spaces=os.path.join(d, f"spaces_{cfg['tag']}.pt"),   # stage1 output
        sens_dir=d,                                           # stage2 outputs: sens_{space}_{tag}.pt
        tag=cfg["tag"],
    )
