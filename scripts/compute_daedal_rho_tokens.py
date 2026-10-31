"""Recover Etoken/Ntoken for DAEDAL/rho-EOS reference-harness results.

The reference harness (dllm_eval) only records accuracy and total wall time in
results_*.json; per-example token counts require re-tokenizing the raw decoded
"resps" text in samples_*.jsonl against the model's own tokenizer (decoded with
skip_special_tokens=False, so <|endoftext|> padding is visible as literal text).

Etoken/Ntoken defined identically to run_benchmark.py's actual_length(): Etoken
is the 1-indexed position of the first EOS token in the generated continuation
(or the full length if no EOS appears); Ntoken is the total generated length
(the realized canvas for that sample, which for DAEDAL varies with expansion).
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

EOS_ID = 126081  # <|endoftext|>, confirmed via results_*.json["eot_token_id"]


def actual_length(token_ids, eos_id):
    for i, t in enumerate(token_ids):
        if t == eos_id:
            return i + 1
    return len(token_ids)


def main():
    samples_path = Path(sys.argv[1])
    tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

    etokens, ntokens, eratios = [], [], []
    with open(samples_path) as f:
        for line in f:
            rec = json.loads(line)
            text = rec["resps"][0][0]
            ids = tok.encode(text, add_special_tokens=False)
            e = actual_length(ids, EOS_ID)
            n_ = len(ids)
            etokens.append(e)
            ntokens.append(n_)
            eratios.append(e / n_ if n_ > 0 else 0.0)

    n = len(etokens)
    print(json.dumps({
        "file": str(samples_path),
        "n": n,
        "mean_etoken": sum(etokens) / n,
        "mean_ntoken": sum(ntokens) / n,
        "eratio_mean_of_ratios": sum(eratios) / n,  # matches run_benchmark.py's per-sample-averaged E_ratio
    }, indent=2))


if __name__ == "__main__":
    main()
