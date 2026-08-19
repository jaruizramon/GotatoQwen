# omp-potato — the GotatoQwen thin client

omp-potato is a fork of **omp** (the terminal LLM client) configured as the
thin client for the GotatoQwen gateway. It runs on the same machine as the
gateway (or any machine that can reach it) and talks to it through the
OpenAI-shaped endpoint at `http://localhost:8090`.

What makes it "potato": a small system prompt that keeps the tiny SLMs
honest — answer directly, no narration, no thinking tags — while the
gateway's delegation machinery (2B brain → stack SLM → tool executor →
verifier) does the real work. omp-potato is **manifest-aware**: the SLMs it
shows come from the stack's manifest (`GET /manifest`, `GET /v1/models`).

## What's in this directory

- `potato-prompt.md` — the system prompt (the fork's personality).
- `run.sh` — the launcher (sets the prompt, execs the omp binary).

## Install

The omp binary itself is not in this repo (182 MB; GitHub's 100 MB limit).
Copy it from the dev box or fetch the upstream omp build, then point the
launcher at it:

```bash
OMP_BIN=/path/to/omp bash run.sh          # or: install to /usr/local/bin/omp
```

The gateway must be running first (see the repo AGENT.md §5): started from
the stack directory, `GOTATO_APPROVE=1` if the tool loop should write files
without per-call prompts.

## How it connects

- omp → `http://localhost:8090/v1/chat/completions` (OpenAI-compatible)
- `GET /manifest` — the launch recipes of the stack's SLMs
- `GET /v1/models` — the stack's SLM list (what omp displays)
- `GET /slms` — the live roster + active slice

## The fork's delta vs upstream omp

1. System prompt tuned for 0.6B-2B SLMs (no narration, no thinking tags).
2. `--no-tools` at the client: tool use is the GATEWAY's job (the routed
   SLM emits `<tool_call>`, the 1.7B executor + 2B verifier run server-side
   — the client stays dumb on purpose).
3. Designed against the delegation contract: the first reply may come from
   the 2B brain, then the session switches to the stack's SLM mid-task
   (visible in the status bar via `/slms`).
