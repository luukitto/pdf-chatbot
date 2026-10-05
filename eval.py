"""Measure retrieval quality: for each question, does one of the top-k retrieved
chunks contain the answer (a short "evidence" phrase copied from the PDF)?
Only retrieval is tested, no LLM calls, so it runs in about a minute.

Run:  python eval.py attention.pdf eval_questions.json
"""
import json
import sys
from pathlib import Path

from rag import get_collection, retrieve

pdf = Path(sys.argv[1]).read_bytes()
cases = json.loads(Path(sys.argv[2]).read_text())

print(f"{len(cases)} questions\n\n| chunk size | hit@1 | hit@4 | hit@8 |\n|---|---|---|---|")
for size in (300, 500, 800, 1500):
    col = get_collection(pdf, size=size, overlap=150)
    # rank of the first chunk containing the evidence (None = not in the top 8)
    ranks = [next((i for i, h in enumerate(retrieve(c["question"], col, k=8)) if c["evidence"] in h["text"]), None)
             for c in cases]
    hit = lambda k: sum(r is not None and r < k for r in ranks) / len(cases)
    print(f"| {size} | {hit(1):.0%} | {hit(4):.0%} | {hit(8):.0%} |")
