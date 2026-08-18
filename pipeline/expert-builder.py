#!/usr/bin/env python3
"""expert-builder.py - the 2-thread worker that slowly analyzes one language.

Spawned by expert-watcher.py (nice 10). Pipeline:
  1. collect  - walk the stack for <lang> files, dedupe, cap -> corpus/<lang>.txt
  2. train    - train_lora.py (manual LoRA, torch, 2 threads, deterministic)
                -> adapters/<lang>/adapter_model.safetensors (peft layout)
  3. convert  - llama.cpp convert_lora_to_gguf.py -> adapters/<lang>.gguf
  4. publish  - index.json {lang: {status: ready, lora, base, samples, ...}}

Storage is cheap: every project keeps its own corpus and adapter on disk and
they are loaded on demand; only the active adapter touches RAM.
"""
import argparse, hashlib, json, os, subprocess, sys, time

TRAINER = "/home/pipo/slm-fleet/train_lora.py"
CONVERT = os.path.join(os.environ.get("GOTATO_LLAMA", "/home/pipo/llama.cpp"), "convert_lora_to_gguf.py")
VENV_PY = os.environ.get("GOTATO_VENV_PY", "/home/pipo/qwen-venv/bin/python")
BASE_GGUF = "Qwen3-0.6B-Q8_0.gguf"          # inference base (GGUF)
HF_BASE = "Qwen/Qwen3-0.6B"                 # training base (HF, via cache)
EXTS = {".py": "python", ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".jsx": "javascript", ".rs": "rust",
        ".go": "go", ".rb": "ruby", ".java": "java", ".kt": "kotlin",
        ".c": "c", ".cpp": "cpp", ".cs": "csharp", ".php": "php"}
MAX_FILES, MAX_TOTAL = 40, 1_000_000


def collect(lang, stack, fleet):
    corpus_dir = os.path.join(fleet, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    seen, files = set(), []
    for root, _, fns in os.walk(stack):
        if ".git" in root or "node_modules" in root or "__pycache__" in root:
            continue
        for fn in fns:
            if EXTS.get(os.path.splitext(fn)[1].lower()) != lang:
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            if not (80 <= len(data) <= 300 * 1024):
                continue
            h = hashlib.sha256(data).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            files.append((p, data))
            if len(files) >= MAX_FILES:
                break
        if len(files) >= MAX_FILES:
            break
    total = sum(len(d) for _, d in files)
    if total > MAX_TOTAL:
        files.sort(key=lambda t: -len(t[1]))
        keep, acc = [], 0
        for p, d in files:
            if acc + len(d) > MAX_TOTAL:
                continue
            keep.append((p, d)); acc += len(d)
        files = keep
    out = os.path.join(corpus_dir, f"{lang}.txt")
    with open(out, "w") as f:
        for p, d in files:
            f.write(f"# ==== {os.path.basename(p)} ====\n")
            f.write(d.decode("utf-8", "replace"))
            f.write("\n")
    return out, len(files), sum(len(d) for _, d in files)


def train(lang, corpus, fleet, epochs=2):
    out_dir = os.path.join(fleet, "adapters", lang)
    cmd = [VENV_PY, TRAINER, "--corpus", corpus, "--out", out_dir,
           "--epochs", str(epochs)]
    env = dict(os.environ, OMP_NUM_THREADS="2")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r.returncode == 0, (r.stderr or r.stdout)[-800:]


def convert(lang, fleet):
    """peft adapter dir -> GGUF LoRA via llama.cpp converter."""
    in_dir = os.path.join(fleet, "adapters", lang)
    out = os.path.join(fleet, "adapters", f"{lang}.gguf")
    hf_cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
    snap = sorted(os.listdir(hf_cache))[0] if os.path.isdir(hf_cache) else None
    base_dir = os.path.join(hf_cache, snap) if snap else None
    cmd = [VENV_PY, CONVERT, "--base", base_dir, "--outfile", out, in_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(out), (r.stderr or r.stdout)[-800:]


def publish(lang, fleet, ok, extra, dt):
    idx_path = os.path.join(fleet, "index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    e = idx.get(lang, {})
    if ok:
        e.update({"status": "ready", "lora": f"adapters/{lang}.gguf",
                  "base": BASE_GGUF, "trained_at": time.time(),
                  "train_seconds": round(dt, 1)})
    else:
        e.update({"status": "error", "error": extra, "error_at": time.time()})
    idx[lang] = e
    tmp = idx_path + ".tmp"
    json.dump(idx, open(tmp, "w"), indent=1)
    os.replace(tmp, idx_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    ap.add_argument("--fleet", default="/home/pipo/slm-fleet")
    ap.add_argument("--stack", default="/home/pipo/stack")
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()

    print(f"[builder:{a.lang}] collect from {a.stack} ...", flush=True)
    corpus, nfiles, nbytes = collect(a.lang, a.stack, a.fleet)
    print(f"[builder:{a.lang}] corpus: {nfiles} files, {nbytes} bytes", flush=True)
    if nfiles == 0:
        publish(a.lang, a.fleet, False, "no corpus files", 0)
        sys.exit(1)
    t0 = time.time()
    print(f"[builder:{a.lang}] training LoRA (2 threads) ...", flush=True)
    ok, err = train(a.lang, corpus, a.fleet, a.epochs)
    if not ok:
        publish(a.lang, a.fleet, False, err, time.time() - t0)
        print(f"[builder:{a.lang}] TRAIN FAILED: {err}", flush=True)
        sys.exit(1)
    print(f"[builder:{a.lang}] converting to GGUF adapter ...", flush=True)
    ok, err = convert(a.lang, a.fleet)
    publish(a.lang, a.fleet, ok, err, time.time() - t0)
    if ok:
        print(f"[builder:{a.lang}] READY -> adapters/{a.lang}.gguf "
              f"in {time.time() - t0:.0f}s", flush=True)
    else:
        print(f"[builder:{a.lang}] CONVERT FAILED: {err}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
