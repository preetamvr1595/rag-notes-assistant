"""Streamlit chat UI.   streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st

from rag_notes.answer import RAG, OpenAICompatBackend, ExtractiveBackend
from rag_notes.index import Index

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="Notes Assistant (RAG)", page_icon="📚", layout="wide")


@st.cache_resource
def load_index():
    p = ROOT / "models" / "index.joblib"
    return Index.load(str(p)) if p.exists() else Index.build(str(ROOT / "data" / "notes"))


index = load_index()
st.title("📚 Notes Assistant")
st.caption("Ask a question — answers come only from the notes in `data/notes`, with sources. Out-of-scope questions are refused.")

with st.sidebar:
    st.header("Settings")
    series_opts = ["all"] + sorted({c.series for c in index.chunks})
    series = st.selectbox("Search in", series_opts)
    backend_kind = st.radio("Answer style", ["Excerpts from my notes (offline)", "LLM via Ollama / OpenAI-compatible"])
    if backend_kind.startswith("LLM"):
        url = st.text_input("Base URL", "http://localhost:11434/v1"); model = st.text_input("Model", "llama3")
        key = st.text_input("API key (not needed for Ollama)", "ollama", type="password")
        backend = OpenAICompatBackend(url, model, key)
    else:
        backend = ExtractiveBackend()
    k = st.slider("Passages retrieved", 3, 8, 5)
    cov = st.slider("Strictness (higher = refuse more)", 0.3, 0.9, 0.65, 0.05)
    st.markdown(f"**{len(index.chunks)}** passages indexed from **{len({c.source for c in index.chunks})}** files")

rag = RAG(index, backend, k=k, min_coverage=cov)
if "history" not in st.session_state:
    st.session_state.history = []
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

q = st.chat_input("e.g. How do I join two DataFrames on a common column?")
if q:
    st.session_state.history.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)
    ans = rag.ask(q, None if series == "all" else series)
    with st.chat_message("assistant"):
        st.markdown(ans.text)
        st.caption(f"backend: {ans.backend} · coverage {ans.confidence:.2f} · {'answered' if ans.answered else 'refused (not in notes)'}")
        with st.expander("Sources" if ans.answered else "Closest passages (not used for an answer)"):
            for s in ans.sources:
                st.markdown(f"**[{s['n']}]** `{s['source']}` › {s['section']}")
    st.session_state.history.append(("assistant", ans.text))
