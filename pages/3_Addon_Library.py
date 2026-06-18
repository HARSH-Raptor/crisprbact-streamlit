import streamlit as st

from crisprbact.addon_library import generate_addon_library

from utils.file_helpers import save_uploaded_file

st.title("Add-on Library")

if "addon_library" not in st.session_state:
    st.session_state.addon_library = None

if st.button("Clear Results"):

    st.session_state.addon_library = None

    st.rerun()

st.markdown(
    "Generate an add-on library to supplement an existing guide library."
)

library_file = st.file_uploader(
    "Upload Existing Library CSV",
    type=["csv"]
)

genome_file = st.file_uploader(
    "Upload GenBank file (.gb or .gbk)",
    type=["gb", "gbk"]
)

guides_per_gene = st.number_input(
    "Target guides per gene",
    min_value=1,
    max_value=20,
    value=3,
    step=1
)

run_button = st.button(
    "Generate Add-on Library",
    type="primary"
)

if run_button:

    if library_file is None:
        st.error("Please upload an existing library CSV.")
        st.stop()

    if genome_file is None:
        st.error("Please upload a GenBank genome.")
        st.stop()

    try:

        library_path = save_uploaded_file(library_file)
        genome_path = save_uploaded_file(genome_file)

        with st.spinner("Generating add-on library..."):

            addon_library = generate_addon_library(
                ref_file=genome_path,
                existing_library=library_path,
                n=guides_per_gene
            )

            st.session_state.addon_library = addon_library

    except Exception as e:

        st.error("Add-on library generation failed")
        st.code(str(e))

if st.session_state.addon_library is not None:

    addon_library = st.session_state.addon_library

    st.success("Add-on library generated successfully")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Guides",
            len(addon_library)
        )

    with col2:
        st.metric(
            "Unique Genes",
            addon_library["locus_tag"].nunique()
            if len(addon_library) > 0 and "locus_tag" in addon_library.columns
            else 0
        )

    csv_data = addon_library.to_csv(index=False)

    st.download_button(
        label="Download Add-on Library CSV",
        data=csv_data,
        file_name="addon_library.csv",
        mime="text/csv"
    )

    st.subheader("Add-on Library Preview")

    st.dataframe(
        addon_library,
        use_container_width=True
    )