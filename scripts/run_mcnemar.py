"""Two-sided paired exact McNemar tests on per-problem correctness.

Default mode runs the tab:pvals tests. --input-dir must hold:
    daedal_{gsm8k,math500}_samples.jsonl, rho_eos_asym_{gsm8k,math500}_samples.jsonl
    daedal_{mbpp,humaneval}.json, rho_eos_asym_{mbpp,humaneval}.json
    smartcrop_{gsm8k,math500,humaneval}_q{90,99}.json, smartcrop_mbpp_q90.json

--pair compares two run_benchmark.py outputs on their shared problems, as in
tab:instance_vs_mean and the eos_free comparisons.

Usage:
    python scripts/run_mcnemar.py --input-dir results/mcnemar_input
    python scripts/run_mcnemar.py --pair math500 A.json B.json
"""
import argparse
import json
import re
from pathlib import Path

from scipy.stats import binomtest

# ── math/gsm8k-style boxed-answer scoring (mirrors metrics/gsm8k.py and
#    metrics/math500.py here and in rho-EOS) ─────────────────────

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]

def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        return s[len("\\boxed "):]
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except (AssertionError, IndexError):
        return s

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str

def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string

def fix_a_slash_b(string):
    if string is None:
        return None
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        float(a); float(b)
        return "\\frac{" + a + "}{" + b + "}"
    except (ValueError, TypeError):
        return string

def strip_string(string):
    if string is None:
        return None
    string = str(string).strip()
    while re.search(r"(\d),(\d{3})", string):
        string = re.sub(r"(\d),(\d{3})", r"\1\2", string)
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("\\%", "").replace("\\%", "")
    string = remove_right_units(string)
    if string.startswith("."):
        string = "0" + string
    if " ." in string:
        string = string.replace(" .", " 0.")
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1].strip()
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string

def is_equiv(str1, str2):
    if str1 is None or str2 is None:
        return False
    s1, s2 = strip_string(str1), strip_string(str2)
    try:
        if abs(float(s1) - float(s2)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    try:
        return strip_string(str1) == strip_string(str2)
    except Exception:
        return str(str1).strip() == str(str2).strip()

def extract_answer(text):
    if "\\boxed" in text or "\\fbox" in text:
        boxed = last_boxed_only_string(text)
        cand = remove_boxed(boxed)
        if cand != text:
            return cand
    m = re.search(r"####\s*([\-\d,./]+)", text)
    if m:
        return m.group(1).replace(",", "")
    return None


def score_dllm_eval_samples(path):
    """{doc_id(int): bool}, replicates metrics/{gsm8k,math500}.py scoring."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = int(item["doc_id"])
            target = str(item.get("target"))
            resps = item.get("resps")
            raw = ""
            if resps and isinstance(resps, list) and resps[0]:
                raw = resps[0][0]
            extracted = None
            boxed = last_boxed_only_string(raw)
            if boxed:
                extracted = remove_boxed(boxed)
            if extracted is None:
                m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
                if m:
                    extracted = m.group(1).strip()
            out[doc_id] = is_equiv(extracted, target)
    return out


def score_smartcrop_gsm8k_math(path):
    """{doc_id(int): bool} from a SmartCrop/Fixed-Length run_benchmark.py output.
    Uses response/gt fields directly (mirrors metrics/math500.py's score_batch,
    used for both gsm8k and math500 in run_benchmark.py), since some files
    (e.g. q99) were never scored server-side."""
    d = json.load(open(path))
    out = {}
    for r in d:
        pred = extract_answer(r["response"])
        gt = extract_answer(r["gt"]) if r["gt"] else None
        out[int(r["doc_id"])] = is_equiv(pred, gt if gt is not None else r["gt"])
    return out


def mcnemar_report(task, name_a, correct_a, name_b, correct_b):
    common = sorted(set(correct_a) & set(correct_b))
    n = len(common)
    a_only = sum(correct_a[i] and not correct_b[i] for i in common)
    b_only = sum(correct_b[i] and not correct_a[i] for i in common)
    both = sum(correct_a[i] and correct_b[i] for i in common)
    neither = sum((not correct_a[i]) and (not correct_b[i]) for i in common)
    acc_a = sum(correct_a[i] for i in common) / n
    acc_b = sum(correct_b[i] for i in common) / n
    discordant = a_only + b_only
    pval = 1.0 if discordant == 0 else binomtest(
        min(a_only, b_only), discordant, 0.5, alternative="two-sided"
    ).pvalue
    print(f"\n=== [{task}] {name_a} (n={n}, acc={acc_a:.4f}) vs {name_b} (acc={acc_b:.4f}) ===")
    print(f"  both correct: {both}   neither: {neither}   {name_a} only: {a_only}   {name_b} only: {b_only}")
    print(f"  discordant total: {discordant}   exact McNemar p-value: {pval:.5f}")
    # Wald 95% CI on the paired accuracy difference; shows what a non-significant p rules out.
    diff = (a_only - b_only) / n
    se = ((a_only + b_only) - (a_only - b_only) ** 2 / n) ** 0.5 / n
    print(f"  paired diff (pp): {100*diff:+.1f}  95% CI [{100*(diff-1.96*se):+.1f}, {100*(diff+1.96*se):+.1f}]")
    return pval


def score_smartcrop_code(path):
    """{task_id(str): bool} from a SmartCrop code-task run_benchmark.py output."""
    d = json.load(open(path))
    return {str(r["doc_id"]): bool(r["correct"]) for r in d}


def load_code_scores(path):
    """{task_id(str): bool} from score_code_tasks.py's output."""
    return {k: bool(v) for k, v in json.load(open(path)).items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=Path("results/mcnemar_input"))
    ap.add_argument("--pair", nargs=3, metavar=("TASK", "A", "B"), help="compare two run_benchmark.py outputs")
    args = ap.parse_args()

    if args.pair:
        task, path_a, path_b = args.pair
        scorer = score_smartcrop_code if task in ("mbpp", "humaneval") else score_smartcrop_gsm8k_math
        mcnemar_report(task, Path(path_a).stem, scorer(path_a), Path(path_b).stem, scorer(path_b))
        raise SystemExit

    D = args.input_dir

    # ── GSM8K ──
    daedal_gsm8k   = score_dllm_eval_samples(f"{D}/daedal_gsm8k_samples.jsonl")
    rho_gsm8k      = score_dllm_eval_samples(f"{D}/rho_eos_asym_gsm8k_samples.jsonl")
    sc90_gsm8k     = score_smartcrop_gsm8k_math(f"{D}/smartcrop_gsm8k_q90.json")
    sc99_gsm8k     = score_smartcrop_gsm8k_math(f"{D}/smartcrop_gsm8k_q99.json")

    mcnemar_report("GSM8K", "SmartCrop-q90", sc90_gsm8k, "DAEDAL", daedal_gsm8k)
    mcnemar_report("GSM8K", "SmartCrop-q90", sc90_gsm8k, "rho-EOS-Asym", rho_gsm8k)
    mcnemar_report("GSM8K", "SmartCrop-q99", sc99_gsm8k, "DAEDAL", daedal_gsm8k)
    mcnemar_report("GSM8K", "SmartCrop-q99", sc99_gsm8k, "rho-EOS-Asym", rho_gsm8k)
    mcnemar_report("GSM8K", "DAEDAL", daedal_gsm8k, "rho-EOS-Asym", rho_gsm8k)

    # ── MATH500 ──
    daedal_math   = score_dllm_eval_samples(f"{D}/daedal_math500_samples.jsonl")
    rho_math      = score_dllm_eval_samples(f"{D}/rho_eos_asym_math500_samples.jsonl")
    sc90_math     = score_smartcrop_gsm8k_math(f"{D}/smartcrop_math500_q90.json")
    sc99_math     = score_smartcrop_gsm8k_math(f"{D}/smartcrop_math500_q99.json")

    mcnemar_report("MATH500", "SmartCrop-q90", sc90_math, "DAEDAL", daedal_math)
    mcnemar_report("MATH500", "SmartCrop-q90", sc90_math, "rho-EOS-Asym", rho_math)
    mcnemar_report("MATH500", "SmartCrop-q99", sc99_math, "DAEDAL", daedal_math)
    mcnemar_report("MATH500", "SmartCrop-q99", sc99_math, "rho-EOS-Asym", rho_math)
    mcnemar_report("MATH500", "DAEDAL", daedal_math, "rho-EOS-Asym", rho_math)

    # ── MBPP ── (SmartCrop only has a q90 run at the canonical n=500)
    daedal_mbpp = load_code_scores(f"{D}/daedal_mbpp.json")
    rho_mbpp    = load_code_scores(f"{D}/rho_eos_asym_mbpp.json")
    sc90_mbpp   = score_smartcrop_code(f"{D}/smartcrop_mbpp_q90.json")

    mcnemar_report("MBPP", "SmartCrop-q90", sc90_mbpp, "DAEDAL", daedal_mbpp)
    mcnemar_report("MBPP", "SmartCrop-q90", sc90_mbpp, "rho-EOS-Asym", rho_mbpp)
    mcnemar_report("MBPP", "DAEDAL", daedal_mbpp, "rho-EOS-Asym", rho_mbpp)

    # ── HumanEval ──
    daedal_he = load_code_scores(f"{D}/daedal_humaneval.json")
    rho_he    = load_code_scores(f"{D}/rho_eos_asym_humaneval.json")
    sc90_he   = score_smartcrop_code(f"{D}/smartcrop_humaneval_q90.json")
    sc99_he   = score_smartcrop_code(f"{D}/smartcrop_humaneval_q99.json")

    mcnemar_report("HumanEval", "SmartCrop-q90", sc90_he, "DAEDAL", daedal_he)
    mcnemar_report("HumanEval", "SmartCrop-q90", sc90_he, "rho-EOS-Asym", rho_he)
    mcnemar_report("HumanEval", "SmartCrop-q99", sc99_he, "DAEDAL", daedal_he)
    mcnemar_report("HumanEval", "SmartCrop-q99", sc99_he, "rho-EOS-Asym", rho_he)
    mcnemar_report("HumanEval", "DAEDAL", daedal_he, "rho-EOS-Asym", rho_he)
