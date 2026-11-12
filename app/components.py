"""Reusable UI helper components for Streamlit."""
import streamlit as st

def render_citation_badge(source: str, section: str) -> None:
    """Renders a formatted badge for cited source notes."""
    st.markdown(
        f'<span class="citation-badge">📌 {source} &gt; {section}</span>',
        unsafe_allow_html=True,
    )

def render_confidence_metric(score: float, threshold: float) -> None:
    """Renders retrieval confidence meter with color coding."""
    delta = score - threshold
    st.metric(label="Relevance Confidence", value=f"{score:.2f}", delta=f"{delta:+.2f}")
