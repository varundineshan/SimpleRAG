# doc_reader.py

import os
from docx import Document


def read_docx(file_path):
    """Extract text from a .docx file."""
    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)


def load_documents(folder_path):
    """
    Load all .docx files from a given folder.

    Returns:
        documents (list): List of document texts.
        doc_names (list): List of corresponding file names.
    """
    documents = []
    doc_names = []
    files = os.listdir(folder_path)
    print("Files in the folder:", files)

    for filename in files:
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            text = read_docx(file_path)
            documents.append(text)
            doc_names.append(filename)

    print(f"Loaded {len(documents)} documents.")
    return documents, doc_names
