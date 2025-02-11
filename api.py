# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import your modules (config, doc_reader, embedding, indexer, retriever, generator)
from config import DOCUMENTS_FOLDER_PATH
from doc_reader import load_documents
from embedding import generate_document_embeddings
from indexer import create_faiss_index
from retriever import retrieve_documents
from generator import generate_answer

app = FastAPI(title="RAG Query API")

# Enable CORS to allow your front-end to make requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to your specific front-end domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your documents, embeddings, and FAISS index at startup.
print("Loading documents and building the index ...")
documents, doc_names = load_documents(DOCUMENTS_FOLDER_PATH)
document_embeddings = generate_document_embeddings(documents)
index = create_faiss_index(document_embeddings)
print("Initialization complete.")

# Define request/response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    referenced_docs: list[str]

@app.post("/query", response_model=QueryResponse)
def query_rag(data: QueryRequest):
    """
    Accept a query from the user, retrieve relevant documents, and generate an answer.
    """
    user_query = data.query
    # Adjust 'k' if necessary; here we use all documents.
    retrieved_context = retrieve_documents(
        query=user_query,
        k=len(documents),
        index=index,
        documents=documents,
        doc_names=doc_names
    )
    answer = generate_answer(user_query, retrieved_context)
    referenced_docs = [doc["doc_name"] for doc in retrieved_context]
    return QueryResponse(answer=answer, referenced_docs=referenced_docs)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
