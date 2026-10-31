"""Backfill the `correct` field on a run_benchmark.py results JSON that was
never scored server-side (e.g. a run predating the scoring step, or one where
scoring was skipped/crashed). Works for gsm8k/math500-style boxed-answer or
"#### N" tasks. Mirrors the extraction/equivalence logic in metrics/math500.py
(used for both gsm8k and math500 by run_benchmark.py's score_all).

Usage:
    python scripts/backfill_boxed_correct.py --path results/gsm8k/some_run.json
"""
import argparse
import json
import re


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


def backfill(path):
    d = json.load(open(path))
    n_correct = 0
    for r in d:
        pred = extract_answer(r["response"])
        gt = extract_answer(r["gt"]) if r["gt"] else None
        c = is_equiv(pred, gt if gt is not None else r["gt"])
        r["correct"] = c
        n_correct += c
    print(f"{path}: n={len(d)} acc={n_correct / len(d):.4f} ({n_correct}/{len(d)})")
    with open(path, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print("wrote scored results back to", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", required=True, help="results JSON to backfill in place")
    args = p.parse_args()
    backfill(args.path)
