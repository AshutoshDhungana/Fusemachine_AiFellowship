# W14: Fine-tuning transformers for support-intent routing

`support_routing.ipynb` is the completed notebook, and it must be run top-to-bottom. The original template is kept as `sample_support_routing.ipynb`.

* **Approach 1** fine-tunes all four encoder candidates (DistilBERT, BERT, RoBERTa, ModernBERT) with AdamW + linear warmup and selects the best on **validation** macro-F1. The winner is evaluated once on the test split.
* **Approach 2** fine-tunes Qwen2.5-0.5B-Instruct with LoRA (r=16) and completion-only loss. It generates the agent name, followed by EOS.
* The best encoder is saved to `artifacts/router_encoder/`. W15 converts it to ONNX and serves it.

Run it on the RTX 4070 (8 GB):

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
jupyter lab support_routing.ipynb   # Kernel → Restart & Run All (~35–50 min in total)
```

After the run, the recommendation cell needs the measured numbers. Claude fills it in from the executed notebook.
