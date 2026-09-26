"""Streamlit UI -> FastAPI backend.  Run: uv run streamlit run ui/app.py"""
import os

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8080")
st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")
st.title("PaperPilot: research assistant for vulnerability-detection papers")

with st.sidebar:
    st.subheader("Backend")
    try:
        h = httpx.get(f"{API}/health", timeout=5).json()
        st.success(f"API up · {h['chunks']} chunks indexed")
        for p in h["providers"]:
            st.caption(f"{'🔴' if p['circuit_open'] else '🟢'} {p['name']} · {p['model']}")
        st.caption(f"Router: {h['router'] or 'not loaded'}")
    except Exception as e:  # noqa: BLE001
        st.error(f"API unreachable: {e}")
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05)
    k = st.slider("retrieved chunks (k)", 1, 8, 5)

tab_chat, tab_agent, tab_route = st.tabs(["💬 Chat (RAG)", "🔎 Research agent", "🧭 Support-intent router"])


def post(path, payload, timeout=180):
    try:
        r = httpx.post(f"{API}{path}", json=payload, timeout=timeout)
        if r.status_code == 429:
            st.warning("Rate limited by the API, please wait a few seconds.")
            return None
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        st.error(f"Request failed: {e}")
        return None


with tab_chat:
    st.session_state.setdefault("history", [])
    for m in st.session_state.history:
        st.chat_message(m["role"]).markdown(m["content"])
    if q := st.chat_input("Ask about the papers…"):
        st.chat_message("user").markdown(q)
        with st.spinner("Thinking…"):
            out = post("/chat", {"question": q, "history": st.session_state.history[-6:],
                                 "temperature": temperature, "top_p": top_p, "k": k})
        if out:
            with st.chat_message("assistant"):
                if out["status"] == "degraded":
                    st.warning("Degraded mode: LLM providers unavailable")
                st.markdown(out["answer"])
                st.caption(f"confidence {out['confidence']:.2f} · {out.get('model', '-')}"
                           f"{' · cached' if out.get('cached') else ''} · {out['latency_ms']} ms")
                with st.expander("Citations / structured output"):
                    st.json(out)
            st.session_state.history += [{"role": "user", "content": q}, {"role": "assistant", "content": out["answer"]}]

with tab_agent:
    st.caption("Multi-step agent: searches, takes notes, verifies its draft with an independent checker, "
               "and may ask you to clarify.")
    q = st.text_area("Research question", "Compare how AI4VA and LineVD use graph neural networks. "
                                          "At what granularity does each predict?")
    c1, c2 = st.columns(2)
    steps = c1.number_input("max steps", 2, 15, 8)
    verify = c2.checkbox("independent verifier", True)
    if st.button("Run agent"):
        with st.spinner("Agent working…"):
            out = post("/agent", {"question": q, "max_steps": int(steps), "verify": verify}, timeout=600)
        if out:
            st.markdown(f"**Status:** `{out['status']}` · {out['n_steps']} steps · {out['total_tokens']} tokens · "
                        f"{out['latency_s']} s")
            st.markdown(out["answer"] or "_(no answer)_")
            st.caption("Citations: " + ", ".join(out["citations"]))
            st.subheader("Trajectory")
            for s in out["steps"]:
                with st.expander(f"{s['step']}. {s['tool']} {'✅' if s['ok'] else '❌'}: {s['reasoning'][:90]}"):
                    st.json({"args": s["args"], "result": s["result"]})

with tab_route:
    st.caption("W14 fine-tuned encoder, exported to ONNX (INT8) and served with micro-batching.")
    msg = st.text_input("Customer message", "My card was charged twice for the same order, I need a refund")
    if st.button("Route"):
        out = post("/route", {"message": msg})
        if out:
            st.metric("Agent", out["agent"], f"confidence {out['confidence']:.2f}")
            st.json(out)
