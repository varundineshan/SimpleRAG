# retriever.py

import numpy as np
from .embedding import get_embedding


def retrieve_documents(query, k, index, documents, doc_names):
    """
    Retrieve the top-k relevant documents for the given query.

    Args:
        query (str): The user query.
        k (int): Number of documents to retrieve.
        index (faiss.Index): The FAISS index.
        documents (list): List of document texts.
        doc_names (list): List of document names.

    Returns:
        List[dict]: Each dictionary contains keys "doc_text" and "doc_name".
    """
    # Embed the query text using the same model as for the documents
    query_embedding = np.array([get_embedding(query)]).astype("float32")

    # Search the FAISS index for the nearest neighbors
    distances, indices = index.search(query_embedding, k)

    # For each retrieved index, create a dictionary with document text and name
    retrieved_docs = [
        {"doc_text": documents[i], "doc_name": doc_names[i]}
        for i in indices[0]
    ]
    return retrieved_docs
