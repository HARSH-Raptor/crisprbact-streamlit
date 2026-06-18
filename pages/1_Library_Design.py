import pandas as pd
import streamlit as st

from crisprbact.library import generate_library
from utils.file_helpers import save_uploaded_file

st.title("Library Design")

if "library" not in st.session_state:
    st.session_state.library = None

if st.button("Clear Results"):
    st.session_state.library = None
    st.rerun()


st.markdown(
    "Generate a CRISPRi guide library from a bacterial GenBank genome."
)

uploaded_file = st.file_uploader(
    "Upload GenBank file (.gb or .gbk)",
    type=["gb", "gbk"]
)

guides_per_gene = st.number_input(
    "Guides per gene",
    min_value=1,
    max_value=20,
    value=3,
    step=1
)

run_button = st.button(
    "Generate Library",
    type="primary"
)

if run_button:

    if uploaded_file is None:
        st.error("Please upload a GenBank file.")
        st.stop()

    try:

        temp_path = save_uploaded_file(uploaded_file)

        with st.spinner("Generating CRISPR library..."):

            library = generate_library(
                ref_file=temp_path,
                n=guides_per_gene
            )

            st.session_state.library = library

    except Exception as e:

        st.error("Library generation failed")
        st.code(str(e))

if st.session_state.library is not None:

    library = st.session_state.library

    st.success("Library generated successfully")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Guides",
            len(library)
        )

    with col2:
        st.metric(
            "Unique Genes",
            library["locus_tag"].nunique()
        )

    csv_data = library.to_csv(index=False)

    st.download_button(
        label="Download Library CSV",
        data=csv_data,
        file_name="crisprbact_library.csv",
        mime="text/csv"
    )

    st.subheader("Library Preview")

    st.dataframe(
        library,
        use_container_width=True
    )