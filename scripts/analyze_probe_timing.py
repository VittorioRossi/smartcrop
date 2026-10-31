"""Summarize probe_timing_*.json: per-task probe share of SmartCrop wall-clock."""

import json
import sys

import numpy as np

EST_PER_STEP = 59.9 / 1024  # fixed-1024 baseline per-step estimate (s)

d = json.load(open(sys.argv[1]))
print(d["meta"])
recs = d["records"]
print("| Task | n | mean L | probe mean (ms) | probe median (ms) | denoise mean (s) | total mean (s) | mean of ratios | ratio of means | denoise/step (ms) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for task in dict.fromkeys(r["task"] for r in recs):
    rs = [r for r in recs if r["task"] == task]
    p = np.array([r["probe_s"] for r in rs])
    dn = np.array([r["denoise_s"] for r in rs])
    t = np.array([r["total_s"] for r in rs])
    L = np.array([r["cropped_len"] for r in rs])
    print(f"| {task} | {len(rs)} | {L.mean():.0f} | {1e3*p.mean():.1f} | {1e3*np.median(p):.1f} | "
          f"{dn.mean():.2f} | {t.mean():.2f} | {100*np.mean(p/t):.3f}% | {100*p.mean()/t.mean():.3f}% | "
          f"{1e3*np.mean(dn/L):.1f} |")
p_all = np.array([r["probe_s"] for r in recs])
print(f"all probe: mean {1e3*p_all.mean():.1f} ms, min {1e3*p_all.min():.1f}, max {1e3*p_all.max():.1f}; estimate {1e3*EST_PER_STEP:.1f} ms")
