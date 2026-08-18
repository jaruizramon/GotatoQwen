#!/usr/bin/env python3
"""expert-watcher.py - watches the stack; spawns 2-thread expert builders.

Thread budget on this 8-thread box:
  - 4 threads: interactive inference (llama-cli -t 4, best measured speed)
  - 2 threads: one expert builder at a time (nice 10, llama-finetune -t 2)
  - 1 thread: this watcher (mostly idle; sleeps between scans)

When the stack gains a language it has never seen, or an existing language's
files change, the watcher re-collects the corpus and spawns a builder that
slowly fine-tunes a per-language expert model. The capability index
(fleet/index.json) is updated on completion; router.py reads it, so the next
request for that language is served by the expert.

Usage:
  expert-watcher.py --stack /path/to/project [--once] [--interval 20]
"""
import argparse, hashlib, json, os, subprocess, sys, time

FLEET = "/home/pipo/slm-fleet"
INDEX = os.path.join(FLEET, "index.json")
BUILDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert-builder.py")

EXTS = {
    ".py": "python", ".pyw": "python", ".pyi": "python",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".rs": "rust",
    ".go": "go", ".rb": "ruby", ".java": "java", ".kt": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".php": "php", ".swift": "swift", ".zig": "zig",
}

def load_index():
    if os.path.exists(INDEX):
        return json.load(open(INDEX))
    return {}

def save_index(idx):
    tmp = INDEX + ".tmp"
    json.dump(idx, open(tmp, "w"), indent=1)
    os.replace(tmp, INDEX)

def scan(stack):
    """Return {lang: {signature, files: [paths]}}."""
    out = {}
    for root, _, files in os.walk(stack):
        if ".git" in root or "node_modules" in root or "__pycache__" in root:
            continue
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            lang = EXTS.get(ext)
            if not lang:
                continue
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if st.st_size < 80 or st.st_size > 400 * 1024:
                continue
            e = out.setdefault(lang, {"files": []})
            e["files"].append(p)
    for lang, e in out.items():
        e["files"].sort()
        h = hashlib.sha256()
        for p in e["files"]:
            h.update(p.encode())
            h.update(str(os.path.getmtime(p)).encode())
        e["signature"] = h.hexdigest()[:16]
    return out

def spawn_builder(lang):
    log = open(os.path.join(FLEET, f"build_{lang}.log"), "a")
    return subprocess.Popen(
        ["nice", "-n", "10", sys.executable, BUILDER, lang,
         "--fleet", FLEET],
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="/home/pipo/stack")
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    print(f"[watcher] watching {a.stack} | index {INDEX}", flush=True)
    while True:
        idx = load_index()
        stack_langs = scan(a.stack)
        running = [p for p in __import__("glob").glob(os.path.join(FLEET, "*.builder.pid"))]
        for lang, e in stack_langs.items():
            entry = idx.get(lang, {})
            if entry.get("status") == "building" and entry.get("pid"):
                # check if the builder is still alive
                try:
                    os.kill(entry["pid"], 0)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    alive = False
                if alive:
                    continue
            if entry.get("status") == "error" and entry.get("error_at", 0) > time.time() - 300:
                continue
            if entry.get("signature") == e["signature"] and entry.get("status") == "ready":
                continue
            # new or changed -> build
            entry.update({"status": "building", "signature": e["signature"],
                          "files": len(e["files"]), "pid": None, "error_at": 0})
            idx[lang] = entry
            save_index(idx)
            p = spawn_builder(lang)
            entry["pid"] = p.pid
            save_index(idx)
            print(f"[watcher] {lang}: {len(e['files'])} files changed -> "
                  f"builder pid {p.pid} (2 threads, nice 10)", flush=True)
        if a.once:
            break
        time.sleep(a.interval)

if __name__ == "__main__":
    main()
