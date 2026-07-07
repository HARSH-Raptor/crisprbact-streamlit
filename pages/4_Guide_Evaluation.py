import streamlit as st
import pandas as pd

from crisprbact.guide_evaluator import evaluate_guides
from utils.file_helpers import save_uploaded_file

st.title("Guide Evaluation")

st.markdown(
    """
Evaluate one or more guide RNA sequences using the original
CRISPRbact scoring and annotation pipeline.
"""
)

if "guide_results" not in st.session_state:
    st.session_state.guide_results = None

if st.button("Clear Results"):

    st.session_state.guide_results = None
    st.rerun()

genome_file = st.file_uploader(
    "Upload GenBank file (.gb or .gbk)",
    type=["gb", "gbk"]
)

guide_text = st.text_area(
    "Guide sequences (one guide per line)",
    height=200,
    placeholder="""GCCTGAAAGCAGAAGACCAG
ATCGATCGATCGATCGATCG
..."""
)

run_button = st.button(
    "Evaluate Guides",
    type="primary"
)

if run_button:

    if genome_file is None:
        st.error("Please upload a GenBank file.")
        st.stop()

    guides = [
        g.strip().upper()
        for g in guide_text.splitlines()
        if g.strip()
    ]

    if len(guides) == 0:
        st.error("Please enter at least one guide.")
        st.stop()

    try:

        genome_path = save_uploaded_file(genome_file)

        with st.spinner("Evaluating guides..."):

            results = evaluate_guides(
                genome_path,
                guides
            )

        st.session_state.guide_results = results

    except Exception as e:

        st.error("Guide evaluation failed")
        st.code(str(e))

if st.session_state.guide_results is not None:

    results = st.session_state.guide_results

    st.success(
        f"Evaluated {len(results)} guide(s)"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Guides Evaluated",
            len(results)
        )

    with col2:

        if "Recommendation" in results.columns:

            st.metric(
                "Recommended Guides",
                (results["Recommendation"] == "⭐ Recommended").sum()
            )

    csv = results.to_csv(index=False)

    st.download_button(
        "Download Results CSV",
        csv,
        "guide_evaluation.csv",
        "text/csv"
    )

    st.subheader("Results")

    st.dataframe(
        results,
        use_container_width=True
    )