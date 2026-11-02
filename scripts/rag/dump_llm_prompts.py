"""Record every prompt the RAG pipeline would send to the LLM, without calling an LLM.

Copy this file to ``D:\\Rag\\scripts\\dump_llm_prompts.py`` and run ``python scripts\\dump_llm_prompts.py``
from ``D:\\Rag`` (Build Guide Phase 9, step 2). Then copy ``eval/llm_prompts.jsonl`` to
``benchmarks/data/rag_prompts.jsonl`` in this repo. That folder is git-ignored ON PURPOSE: the
prompts contain text from arXiv papers, so only this script is published.

If ``run_config`` or ``load_questions`` has changed, read ``src/eval.py: main()`` and call what it calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.generator as generator          # generator.py did `from src.llm import generate`
from src import eval as rag_eval
from src.llm import LLMResult

records = []


def recording_generate(prompt, system=None, schema=None):
    records.append({"system": system, "prompt": prompt, "uses_schema": schema is not None})
    return LLMResult(text="This is a placeholder answer. [1]", input_tokens=0, output_tokens=0,
                     latency_s=0.0)


generator.generate = recording_generate     # patch the name generator.py actually calls (P41)

for path in ["eval/questions.json", "eval/heldout_questions.json"]:
    rag_eval.run_config("router", rag_eval.load_questions(Path(path)))

unique = list({json.dumps(r, sort_keys=True): r for r in records}.values())
Path("eval/llm_prompts.jsonl").write_text("\n".join(json.dumps(r) for r in unique), encoding="utf-8")
print(f"wrote {len(unique)} prompts")
