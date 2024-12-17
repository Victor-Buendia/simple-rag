import pymupdf
import pymupdf4llm

from streamlit.runtime.uploaded_file_manager import UploadedFile


def extract_markdown_from_pdf(uploaded_file: UploadedFile):
    document = pymupdf.Document(filename=uploaded_file, stream=uploaded_file.getvalue())
    return pymupdf4llm.to_markdown(document)
