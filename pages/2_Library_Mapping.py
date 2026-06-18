import pandas as pd
import streamlit as st

from Bio import SeqIO

from crisprbact.map_library import map_library
from crisprbact.visualize import generate_map_report

from utils.file_helpers import save_uploaded_file

st.title("Library Mapping")

if "mapped_library" not in st.session_state:
    st.session_state.mapped_library = None

if "mapping_report" not in st.session_state:
    st.session_state.mapping_report = None

if st.button("Clear Results"):

    st.session_state.mapped_library = None
    st.session_state.mapping_report = None

    st.rerun()

st.markdown(
    "Map an existing guide library against a bacterial genome."
)

library_file = st.file_uploader(
    "Upload Library CSV",
    type=["csv"]
)

genome_file = st.file_uploader(
    "Upload GenBank file (.gb or .gbk)",
    type=["gb", "gbk"]
)

run_button = st.button(
    "Map Library",
    type="primary"
)

if run_button:

    if library_file is None:
        st.error("Please upload a library CSV.")
        st.stop()

    if genome_file is None:
        st.error("Please upload a GenBank genome.")
        st.stop()

    try:

        library_path = save_uploaded_file(library_file)
        genome_path = save_uploaded_file(genome_file)

        with st.spinner("Mapping library..."):

            mapped_library = map_library(
                guides=library_path,
                ref_file=genome_path
            )

            records = list(
                SeqIO.parse(
                    genome_path,
                    "genbank"
                )
            )

            report_path = "temp/report.html"

            generate_map_report(
                mapped_library,
                records,
                report_path
            )

            with open(report_path, "r", encoding="utf-8") as f:
                report_html = f.read()

            st.session_state.mapped_library = mapped_library
            st.session_state.mapping_report = report_html

    except Exception as e:

        st.error("Library mapping failed")
        st.code(str(e))

if st.session_state.mapped_library is not None:

    mapped_library = st.session_state.mapped_library

    st.success("Library mapped successfully")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Rows",
            len(mapped_library)
        )

    with col2:
        st.metric(
            "Unique Guides",
            mapped_library["guide"].nunique()
        )

    csv_data = mapped_library.to_csv(index=False)

    st.download_button(
        label="Download Mapping CSV",
        data=csv_data,
        file_name="mapped_library.csv",
        mime="text/csv"
    )

    if st.session_state.mapping_report is not None:

        st.download_button(
            label="Download HTML Report",
            data=st.session_state.mapping_report,
            file_name="mapping_report.html",
            mime="text/html"
        )

    st.subheader("Mapping Preview")

    st.dataframe(
        mapped_library,
        use_container_width=True
    )