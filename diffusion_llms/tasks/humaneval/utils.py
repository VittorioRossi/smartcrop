import re

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

