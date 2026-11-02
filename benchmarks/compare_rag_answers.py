"""Compare two RAG to-grade CSVs (Build Guide Phase 15 C): exact-match rate after whitespace
normalization, and the list of questions whose answers differ.

  python benchmarks/compare_rag_answers.py ollama_to_grade.csv turingserve_to_grade.csv
"""

from __future__ import annotations

import csv
import re
import sys


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    key = next(k for k in rows[0] if "question" in k.lower())
    ans = next((k for k in rows[0] if "answer" in k.lower()), None)
    if ans is None:
        raise SystemExit(f"{path}: no answer column in {list(rows[0])}")
    return {norm(r[key]): norm(r[ans]) for r in rows}


def main(a: str, b: str) -> int:
    A, B = load(a), load(b)
    common = sorted(set(A) & set(B))
    same = [q for q in common if A[q] == B[q]]
    print(f"{len(common)} common questions; exact match {len(same)}/{len(common)} "
          f"= {len(same) / max(1, len(common)):.1%}")
    for q in common:
        if A[q] != B[q]:
            print(f"\nQ: {q}\n  A: {A[q][:300]}\n  B: {B[q][:300]}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
