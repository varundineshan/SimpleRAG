import numpy as np
from .embedding import get_embedding

# Set a threshold for relevance (adjust based on testing)
SIMILARITY_THRESHOLD = 0.6  # Adjust this value based on performance

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
    query_embedding = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    retrieved_docs = []
    for i in range(len(indices[0])):
        doc_index = indices[0][i]
        similarity_score = 1 / (1 + distances[0][i])  # Convert L2 distance to similarity

        if similarity_score >= SIMILARITY_THRESHOLD:
            retrieved_docs.append({
                "doc_text": documents[doc_index],
                "doc_name": doc_names[doc_index]
            })

    return retrieved_docs
