"""Build the regression reference set from the best prompt version.

  uv run python -m mlops.make_golden --config mlops/configs/v3.yaml
Writes mlops/regression/golden.jsonl. REVIEW IT BY HAND before committing: golden answers are
"approved" responses, so fix or delete any entry you would not sign off on.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.agent.loop import ResearchAgent  # noqa: E402
from assistant.config import get_settings  # noqa: E402
from assistant.llm.client import ResilientLLM  # noqa: E402
from assistant.rag.store import VectorStore  # noqa: E402
from assistant.tools import ToolBox  # noqa: E402

REG_IDS = ["q01_ai4va_repr", "q02_vudenc_types", "q03_tencent_fp", "q04_malcodeai_model", "q05_cyberllm_coverage",
           "q06_gap_generalization", "q08_compare_ai4va_linevd", "q09_compare_cyberllm_malcodeai",
           "q12_fp_vs_cyberllm_precision", "q15_out_of_corpus"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    s = get_settings()
    s.top_k = cfg["top_k"]
    cases = {c["id"]: c for c in yaml.safe_load((ROOT / "eval/cases.yaml").read_text(encoding="utf-8"))["cases"]}
    agent = ResearchAgent(ResilientLLM(s), ToolBox(VectorStore(s), None), prompt=cfg["prompt"],
                          max_steps=cfg["max_steps"], temperature=cfg["temperature"], verify_answers=cfg["verify"])
    out = ROOT / "mlops/regression/golden.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for cid in REG_IDS:
            r = await agent.run(cases[cid]["query"])
            f.write(json.dumps({"id": cid, "query": cases[cid]["query"], "reference": r.answer,
                                "reference_status": r.status, "reference_citations": r.citations,
                                "source_version": cfg["version"]}, ensure_ascii=False) + "\n")
            print(cid, r.status)
    print(f"wrote {out}: review each reference answer before committing")


if __name__ == "__main__":
    asyncio.run(main())
