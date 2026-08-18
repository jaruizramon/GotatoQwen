#!/usr/bin/env python3
"""router.py - the "indexed model": capability index + language detection + delegation.

Tier architecture:
  Tier 0 (API)   : big LLM (Qwen3.8-27B via cloud) does the scaffold & planning.
  Tier 1 (THIS)  : local router. Maintains a capability index (language -> model,
                   quant, measured tok/s, context), detects the request language,
                   and delegates to the matching local micro-expert SLM. Escalates
                   to the API tier when the local fleet can't handle the task.
  Tier 2 (local) : per-language micro-experts (GGUF SLMs or base+LoRA).

Detection for v1 is lexical (extensions + keyword fingerprints) - zero extra
RAM. A trained classifier over activation features replaces it in v2.

Usage:
  router.py detect <file-or-dir>          # what language is this?
  router.py run <file-or-prompt> [--gen N]  # detect + delegate to local SLM
  router.py bench [--lang python]         # measure tok/s per fleet member
  router.py --config fleet.yaml
"""
import argparse, json, os, re, subprocess, sys, time

LLAMA_CLI = os.path.join(os.environ.get("GOTATO_LLAMA_BIN", "/home/pipo/llama.cpp/build/bin"), "llama-cli")
FLEET_DIR = os.environ.get("GOTATO_FLEET", "/home/pipo/slm-fleet")
INDEX = os.path.join(FLEET_DIR, "index.json")


def expert_model_for(lang):
    """If the watcher/builder has produced a ready expert for this language,
    return (base_model, lora_adapter) paths, else (None, None) to fall back
    to the generalist fleet."""
    try:
        idx = json.load(open(INDEX))
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None
    e = idx.get(lang)
    if e and e.get("status") == "ready" and e.get("lora"):
        base = os.path.join(FLEET_DIR, e.get("base", "") or "")
        lora = os.path.join(FLEET_DIR, e["lora"])
        if os.path.exists(lora) and os.path.exists(base):
            return base, lora
    return None, None

# ---- capability index (v1: one model per language; LoRA swap in v2) --------
FLEET = {
    "python": {
        "model": "/home/pipo/slm-fleet/Qwen3.5-2B-Q4_K_M.gguf",
        "q4_gb": 1.28, "toks_per_s": None, "note": "2B generalist (LoRA: python)",
    },
    "typescript": {
        "model": "/home/pipo/slm-fleet/Qwen3.5-2B-Q4_K_M.gguf",
        "q4_gb": 1.28, "toks_per_s": None, "note": "2B generalist (LoRA: ts)",
    },
    "javascript": {
        "model": "/home/pipo/slm-fleet/Qwen3.5-4B-Q4_K_M.gguf",
        "q4_gb": 2.74, "toks_per_s": None, "note": "4B generalist (LoRA: js)",
    },
    "rust": {
        "model": "/home/pipo/slm-fleet/Qwen3.5-4B-Q4_K_M.gguf",
        "q4_gb": 2.74, "toks_per_s": None, "note": "4B generalist (LoRA: rust)",
    },
    "default": {
        "model": "/home/pipo/slm-fleet/Qwen3.5-4B-Q4_K_M.gguf",
        "q4_gb": 2.74, "toks_per_s": None, "note": "fallback",
    },
}

# ---- lexical fingerprints: file extensions + strong code signals ----------
EXT2LANG = {
    ".py": "python", ".pyw": "python", ".pyi": "python", ".toml": "python",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".rs": "rust", ".cargo": "rust",
    ".json": None, ".md": None, ".txt": None, ".yaml": None, ".yml": None,
}
SIGNALS = {
    "python": [r"^\s*(def |class |import |from |async def )", r"print\(.*\)", r"self\.",
               r"if __name__", r":\s*$", r"#.*$"],
    "typescript": [r"(interface |type |const .*: |function .*: |: string|: number)",
                   r"import .* from ['\"]", r"export (default )?(function|class|const)"],
    "javascript": [r"(function |const .* = \(.*\) =>|require\(|module\.exports)",
                   r"console\.log\(|=>\s*\{"],
    "rust": [r"^\s*(fn |let mut |impl |pub |use )", r"println!\(|unwrap\(\)|\.await",
             r"-> \w+ \{", r"\w+::\w+\(\)"],
}

def detect_language(text, path=None):
    """Return (lang, confidence). Extension first, then keyword fingerprints."""
    if path:
        ext = os.path.splitext(path)[1].lower()
        if ext in EXT2LANG and EXT2LANG[ext]:
            return EXT2LANG[ext], 0.9
    scores = {lang: 0 for lang in SIGNALS}
    head = text[:4000]
    for lang, pats in SIGNALS.items():
        for p in pats:
            if re.search(p, head, re.M):
                scores[lang] += 1
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    if total == 0:
        return "default", 0.1
    return best, scores[best] / total

def build_prompt(lang, task):
    if lang == "default":
        return task
    return (f"You are a senior {lang} engineer on a laptop. The project scaffold "
            f"was planned by an architect model; your job is to implement the "
            f"requested piece of {lang} code, idiomatic, minimal, no comments "
            f"explaining yourself.\n\n{task}")

def run_local(lang, prompt, gen=128, threads=4):
    base, lora = expert_model_for(lang)
    if base:
        model, used = base, "expert+lora"
    else:
        entry = FLEET.get(lang, FLEET["default"])
        model, used = entry["model"], "generalist"
    if not os.path.exists(model):
        return None, f"model missing: {model}", used
    cmd = [LLAMA_CLI, "-m", model, "-p", prompt, "-n", str(gen), "-t", str(threads),
           "-c", "4096", "--temp", "0.3", "--log-disable", "-st"]
    if lora:
        cmd += ["--lora", lora]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "timeout", used
    dt = time.time() - t0
    out = r.stdout
    return out, dt, used

def bench(lang=None, gen=32):
    langs = [lang] if lang else list(FLEET)
    for l in langs:
        entry = FLEET[l]
        if not os.path.exists(entry["model"]):
            print(f"{l:12s} model not downloaded yet"); continue
        prompt = build_prompt(l, "Write a function that greets the user.")
        out, dt, used = run_local(l, prompt, gen=gen)
        if out is None:
            print(f"{l:12s} FAILED: {dt}"); continue
        tps = gen / dt
        entry["toks_per_s"] = round(tps, 1)
        print(f"{l:12s} {tps:5.1f} tok/s  ({dt:.1f}s for {gen} tokens)  "
              f"model={os.path.basename(entry['model'])}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["detect", "run", "bench"])
    ap.add_argument("target", nargs="?", default=None)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--lang", default=None)
    a = ap.parse_args()

    if a.mode == "bench":
        bench(a.lang); sys.exit(0)

    text = ""
    path = None
    if a.target and os.path.exists(a.target):
        path = a.target
        with open(a.target, "r", errors="ignore") as f:
            text = f.read()
    elif a.target:
        text = a.target

    lang, conf = detect_language(text, path)
    base, _ = expert_model_for(lang)
    print(f"[router] lang={lang} conf={conf:.2f} "
          f"-> {os.path.basename(base or FLEET.get(lang, FLEET['default'])['model'])}")

    if a.mode == "detect":
        sys.exit(0)

    prompt = build_prompt(lang, text if text else "Write a hello-world function.")
    out, dt, used = run_local(lang, prompt, gen=a.gen)
    if out is None:
        print(f"[router] FAILED: {dt}"); sys.exit(1)
    body = out.strip().split("\n")
    body = "\n".join(line for line in body if not line.startswith("> "))
    print(body)
    print(f"\n[router] [{used}] {a.gen} tokens in {dt:.1f}s ({a.gen/dt:.1f} tok/s)")
