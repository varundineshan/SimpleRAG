# app/api.py
from .doc_reader import load_documents
from .embedding import generate_document_embeddings
from .indexer import create_faiss_index
from .retriever import retrieve_documents
from .generator import generate_answer

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import traceback
from pathlib import Path

# Import your actual functions from the helper modules.
# Adjust these imports if your files are organized differently.



# -------------------------------------------------------------------
# Setup: Compute project root and static folder path.
# -------------------------------------------------------------------
# Since this file is in "app/", the project root is one level up.
BASE_DIR = Path(__file__).resolve().parent.parent

# Absolute path to the "static" folder.
STATIC_DIR = BASE_DIR / "static"
if not STATIC_DIR.exists():
    raise RuntimeError(f"Static directory '{STATIC_DIR}' does not exist")

# -------------------------------------------------------------------
# Create the FastAPI application.
# -------------------------------------------------------------------
app = FastAPI(title="RAG Query API")

# Enable CORS (adjust allowed origins for production as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your actual front-end domain in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optionally, mount additional static files (e.g., for CSS, images) at "/static".
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Define a GET endpoint for "/" that returns index.html.
@app.get("/")
def read_index():
    index_file = STATIC_DIR / "index.html"
    return FileResponse(str(index_file))

# -------------------------------------------------------------------
# Initialize your documents and build the FAISS index.
# -------------------------------------------------------------------
# Replace DOCUMENTS_FOLDER_PATH with your actual documents folder if needed.
DOCUMENTS_FOLDER_PATH = str(BASE_DIR / "documents")  # Change as needed.
documents, doc_names = load_documents(DOCUMENTS_FOLDER_PATH)
document_embeddings = generate_document_embeddings(documents)
index = create_faiss_index(document_embeddings)

# -------------------------------------------------------------------
# Define Pydantic models for the API request and response.
# -------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    referenced_docs: list[str]

# -------------------------------------------------------------------
# Define the /query endpoint.
# -------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
def query_rag(data: QueryRequest):
    print("Received query:", data.query)
    try:
        user_query = data.query
        # Retrieve documents based on the query using your actual function.
        retrieved_context = retrieve_documents(
            query=user_query,
            k=len(documents),
            index=index,
            documents=documents,
            doc_names=doc_names
        )
        print("Retrieved context:", retrieved_context)
        # Generate an answer using your actual answer generation function.
        answer = generate_answer(user_query, retrieved_context)
        print("Generated answer:", answer)
        # Extract document names to return in the response.
        referenced_docs = [doc["doc_name"] for doc in retrieved_context]
        return QueryResponse(answer=answer, referenced_docs=referenced_docs)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# -------------------------------------------------------------------
# Run the app when executing this file directly.
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
