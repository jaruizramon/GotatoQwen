#!/usr/bin/env python3
"""distill.py - use the (now runnable) 27B as a local distillation oracle.

On the 40GB laptop the original Qwen3.8-27B runs at ~2-3 tok/s - too slow for
interactive use, exactly right for generating exemplars. For each language in
the stack, it produces N exemplar solutions to the task prompts, appends them
to the project corpus, and retrains the expert (the "extend" loop: storage is
cheap, every project grows its expert).

Usage (on the laptop):
  python3 distill.py --lang python --tasks tasks/python.txt --n 8
  # then: python3 train_lora.py --corpus corpus/python.txt --out adapters/python \
  #        --base-model unsloth/Qwen3.5-2B   (40GB box: 2B base, real expertise)
"""
import argparse, json, os, subprocess, sys, time

LLAMA_CLI = "llama.cpp/build/bin/llama-cli"
ORACLE = "models/qwen3.8-27b-q4km.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--tasks", required=True, help="file with task prompts, one per line")
    ap.add_argument("--n", type=int, default=8, help="exemplars per task")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--oracle", default=ORACLE)
    ap.add_argument("--corpus-out", required=True)
    a = ap.parse_args()

    tasks = [l for l in open(a.tasks) if l.strip()]
    print(f"[distill] oracle={a.oracle} lang={a.lang} tasks={len(tasks)} x{a.n}",
          flush=True)

    examples = []
    for task in tasks:
        for i in range(a.n):
            prompt = (f"Write idiomatic {a.lang} code. Task: {task.strip()}\n"
                      f"Respond with only the code, no explanation.")
            cmd = [LLAMA_CLI, "-m", a.oracle, "-p", prompt, "-n", str(a.gen),
                   "-t", "4", "-c", "4096", "--temp", "0.7", "--log-disable", "-st"]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            out = r.stdout
            # strip the model's thinking block and timing lines
            if "<think>" in out and "</think>" in out:
                out = out.split("</think>", 1)[1]
            code = "\n".join(l for l in out.splitlines()
                             if not l.startswith("> ") and "t/s" not in l)
            if len(code.strip()) < 40:
                print(f"  [skip] short/empty exemplar for: {task.strip()[:40]}")
                continue
            examples.append(f"# ---- oracle exemplar: {task.strip()} ----\n{code}")
            print(f"  [{i + 1}/{a.n}] {task.strip()[:40]}... "
                  f"({len(code)} chars, {time.time() - t0:.0f}s)", flush=True)

    with open(a.corpus_out, "a") as f:
        f.write("\n\n".join(examples) + "\n")
    print(f"[distill] appended {len(examples)} exemplars -> {a.corpus_out}", flush=True)
    print(f"[distill] next: train_lora.py --corpus {a.corpus_out} "
          f"--out adapters/{a.lang} --base-model <2B-or-4B-HF-id>", flush=True)


if __name__ == "__main__":
    main()
