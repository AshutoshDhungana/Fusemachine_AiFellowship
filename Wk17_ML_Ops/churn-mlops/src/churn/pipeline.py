"""End-to-end workflow: data -> training -> tracking -> registry -> (serving) -> monitoring -> retraining if needed.

  uv run churn-pipeline            (or: uv run python -m churn.pipeline)
"""
from __future__ import annotations

from .data import load_clean, reference_current_split
from .drift import run as drift_check
from .train import register_best, train_all


def main():
    ref, _ = reference_current_split(load_clean())
    table = train_all(ref, "initial")
    print(table.drop(columns=["run_id"]).to_string(index=False))
    info = register_best(table)
    print(f"registered {info['model_name']} v{info['version']} ({info['run']}) -> Production")
    s = drift_check(inject=True)
    if s["decision"].startswith("RETRAIN"):
        from .retrain import main as retrain
        retrain()


if __name__ == "__main__":
    main()
