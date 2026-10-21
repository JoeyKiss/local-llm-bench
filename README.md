# Local LLM Bench

A small, dependency-free benchmark for comparing **local LLMs served by [Ollama](https://ollama.com)** on
your own hardware. For every model it measures

- **generation and prompt-processing speed** (tokens/s),
- **GPU fit**: how much of the model actually sits in VRAM versus spilling into system RAM,
- **quality** on a small, auto-graded task set (German and English): arithmetic and logic, grammar and
  spelling, JSON extraction, strict instruction following, Python coding with unit tests, and needle
  retrieval from a ~5k-token text.

## Why

Leaderboards say little about what runs well on *one* machine. A model that scores best can still be
unusable if it only half fits into 16 GB of VRAM. This bench puts speed, memory fit and answer quality
into one table, so the trade-off is visible, for example "bigger model, 6x slower, same score".

## Features

- Stdlib-only Python, one file ([bench.py](bench.py)); talks to Ollama's HTTP API.
- Reports load time, model size, share of the model on the GPU, generation tok/s and prompt tok/s.
- Quality tasks run at temperature 0, fixed seed and `num_ctx` 8192, so reruns are comparable.
- Graders are programmatic, not LLM-judged: number parsing that understands German and English formats
  (`2.318` / `2,318` / `1.234,56`), JSON schema checks, and unit tests run against generated Python in
  an isolated subprocess (`python -I`, temp directory, 20 s timeout).
- Two task sets: `base` (13 tasks) and `extra` (20 tasks); `all` runs both.
- Optional `--think` mode lets reasoning models think before answering.
- Partial results are written after each model, so a hung model does not lose the run.

## Requirements

- Python 3 (standard library only; developed on 3.13)
- A running Ollama server on `localhost:11434` with the models you want to test already pulled

## Usage

```bash
python -I bench.py                              # default model list (DEFAULT_MODELS in bench.py)
python -I bench.py qwen3.5:9b llama3.1:8b       # specific models
python -I bench.py --set extra qwen3.5:9b       # task set: base (default), extra, all
python -I bench.py --think qwen3.5:9b           # let thinking models reason first
```

Results are written to `results/<timestamp>.md` (table) and `results/<timestamp>.json` (including each
model's raw answers, which is worth reading before trusting a small score difference).

Note: the coding tasks execute model-generated Python on your machine. The subprocess is isolated from
the project folder but is not a security sandbox, so only benchmark models you are comfortable running.

For clean numbers, close games and other GPU-heavy programs first: a run with a game open and a
parallel model download measured roughly half the real speed.

## Example output

Base set (13 tasks), no thinking, on an RTX 5080 (16 GB), Ryzen 7 9800X3D, 32 GB RAM:

| Model | Score | Gen tok/s | Prompt tok/s | Size GB | % on GPU |
|---|---|---|---|---|---|
| qwen3.6:35b-a3b | 13/13 | 82.8 | 1555 | 22.5 | 59 |
| qwen3.8:27b | 11/13 | 13.1 | 1079 | 18.2 | 68 |
| qwen3.5:9b | 11/13 | 117.1 | 5915 | 5.7 | 100 |
| Moonlight-16B-A3B-Instruct (Q4_K_M) | 7/13 | 250.7 | 6899 | 10.7 | 100 |
| qwen2.5:14b | 10/13 | 76.5 | 3514 | 10.4 | 100 |
| qwen2.5:7b | 10/13 | 144.0 | 7491 | 5.1 | 100 |
| llama3.1:8b | 8/13 | 142.8 | 7074 | 5.9 | 100 |

What the table shows: the 27B model, which only 68 % fits on the GPU, generates at 13 tok/s while the
9B model that fits completely runs at 117 tok/s with the same score. The mixture-of-experts model
(35B total, ~3B active) stays fast (83 tok/s) even though 41 % of it lives in system RAM.

Full per-task tables are in [results/](results/). The task set is small, so a one- or two-point gap is
noise. The first base run predates a few later grader fixes (German number parsing, looser dative and
Konjunktiv checks), so re-running with the current code can shift individual scores.

## Project layout

```
bench.py     benchmark, task definitions and graders
results/     example runs (.md summary + .json with raw answers)
```

## License

[MIT](LICENSE)
