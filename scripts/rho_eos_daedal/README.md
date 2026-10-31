# DAEDAL / ρ-EOS reproduction scripts

The eval and Slurm scripts behind the DAEDAL and ρ-EOS-Asym baseline numbers,
run out of a clone of the authors' own reference implementation:

    https://github.com/yjyddq/rho-EOS
    pinned commit: 69992caff275e5cd4c96f654aad633fca69f3f75

To run them, clone `rho-EOS` at the commit above, copy `*.sh` into
`rho-EOS/scripts/` and `*.slurm` into the `rho-EOS` root, and submit from there.

## Files

- `eval_baseline_init64_bs1.sh`, `baseline_init64_bs1.slurm` — Table 1 right
  block: the authors' L_init=64 at batch size 1, threshold decoding.
- `eval_daedal_1024.sh` / `eval_rho_eos_asym_1024.sh` — original 4-task
  (gsm8k, math500, mbpp, humaneval) sweep scripts, unmodified from first use.
- `eval_daedal_1024_codeonly.sh` / `eval_rho_eos_asym_1024_codeonly.sh` —
  mbpp+humaneval-only variants, used to resume a sweep after gsm8k/math500
  had already completed, without re-running them.
- `daedal_full.slurm`, `daedal_codeonly.slurm`, `rho_eos_asym_codeonly.slurm`
  — SLURM wrappers (1 GPU, 24h time limit).

## Known fixes baked into the `codeonly` variants

1. **Isolated HF datasets cache** (`HF_DATASETS_CACHE=$HOME/.cache/huggingface_rho_eos/datasets`).
   The `rho-EOS` conda env's older `datasets` library can't parse the
   `DatasetInfo` schema written by the `diffusion_llms` env's newer
   `datasets` when both share the default `~/.cache/huggingface` path —
   raises `TypeError: must be called with a dataclass type or instance`
   in `datasets/features.py`. Giving the `rho-EOS` env its own cache dir
   avoids the version-skew clash entirely. This fix was only applied to
   the `_codeonly` scripts; if `eval_daedal_1024.sh` /
   `eval_rho_eos_asym_1024.sh` ever need a full rerun from scratch, add
   the same env var there first or they will hit the same crash on mbpp.

2. **`--batch_size 4` (down from 8)** in `eval_rho_eos_asym_1024_codeonly.sh`.
   The original batch_size=8 run hit a CUDA OOM 480/500 samples into
   mbpp (process alone using 80GB of a 93GB H100) — ρ-EOS's canvas
   expansion grows the KV-cache per-sequence across a batch, and long
   mbpp completions pushed memory over the edge near the end of the task.
   `eval_daedal_1024*.sh` did not hit this and is left at batch_size=8.

## Accuracy readout gotcha

The harness's own `results_*.json` always reports `exact_match,none: 0.0`
for gsm8k/math500 — that's the harness's default answer-parsing filter,
which doesn't understand the `<answer>\boxed{...}</answer>` format these
scripts prompt for. The real score is printed only to stdout by the
second-pass `python metrics/{task}.py --res_path ...` step baked into
each eval script, as `Accuracy: XX.XX%` — grep the run's `.log` file for
that string, never read the harness's own `results.json` for accuracy.
