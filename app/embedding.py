# embedding.py

import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
from .config import EMBEDDING_MODEL
import os
load_dotenv("app/.env")
# Create a client for embedding calls (using a fixed API version)
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2023-05-15",
)

def get_embedding(text, model=EMBEDDING_MODEL):
    """Generate an embedding for the given text using OpenAI's ada-002 model."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    embedding = response.data[0].embedding
    return embedding


def generate_document_embeddings(documents):
    """
    Generate embeddings for a list of documents.

    Returns:
        A NumPy array of embeddings with dtype float32.
    """
    document_embeddings = []
    for doc in documents:
        emb = get_embedding(doc)
        document_embeddings.append(emb)

    # Convert to a NumPy array with dtype float32 (required by FAISS)
    document_embeddings = np.array(document_embeddings).astype("float32")
    return document_embeddings
