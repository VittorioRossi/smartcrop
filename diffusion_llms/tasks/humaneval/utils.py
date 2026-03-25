import re
import evaluate as hf_evaluate

_code_eval = hf_evaluate.load("code_eval")


def build_predictions(resps: list, docs: list) -> list:
    """Prepend the function signature (prompt) to each model completion."""
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def pass_at_k(references: list, predictions: list, k=None):
    """Compute pass@k via HuggingFace code_eval (called per-sample by lm_eval)."""
    if k is None:
        k = [1]
    if isinstance(k, int):
        k = [k]
    result, _ = _code_eval.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return result


def humaneval_postprocess_v2(resps, docs):
    processed_resps = []
    for r in resps:
        # resps is a list of lists (one list per doc, containing n generations)
        # We process each generation in the list
        processed_doc_resps = []
        for text in r:
            blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
            if len(blocks) >= 1:
                text = blocks[0]
            processed_doc_resps.append(text.lstrip())
        processed_resps.append(processed_doc_resps)
    return processed_resps

