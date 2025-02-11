# indexer.py

import faiss

def create_faiss_index(embeddings):
    """
    Create a FAISS index (using L2 distance) from the given embeddings.

    Args:
        embeddings (np.array): NumPy array of document embeddings.

    Returns:
        index: The FAISS index with the embeddings added.
    """
    # Determine the dimensionality from one of the embeddings
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    # Add the document embeddings to the index
    index.add(embeddings)
    print(f"Number of documents indexed: {index.ntotal}")
    return index
