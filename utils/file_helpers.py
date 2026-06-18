import os


def save_uploaded_file(uploaded_file, temp_dir="temp"):
    """
    Save uploaded Streamlit file to disk.
    Returns path of saved file.
    """

    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path