# main.py

from config import DOCUMENTS_FOLDER_PATH
from doc_reader import load_documents
from embedding import generate_document_embeddings
#from indexer import create_faiss_index
#from retriever import retrieve_documents
#from generator import generate_answer
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
# Access the secrets
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
def main():
    # Load documents and their names from the folder
    documents, doc_names = load_documents(DOCUMENTS_FOLDER_PATH)

    # Generate embeddings for the loaded documents
    document_embeddings = generate_document_embeddings(documents)
    print(document_embeddings)

    # Create a FAISS index from the embeddings
    #index = create_faiss_index(document_embeddings)

    # Define the user query
    user_query = (
        "how much will it cost to create rag model with 30 page document also mention "
        "what technologies are included in the SRS document"
    )

    # Retrieve the top relevant documents (using all documents in this case)
    '''retrieved_context = retrieve_documents(
        query=user_query,
        k=len(documents),
        index=index,
        documents=documents,
        doc_names=doc_names
    )'''

    # Generate an answer using the retrieved context
    #answer = generate_answer(user_query, retrieved_context)

    print("Answer:")
    #print(answer)

    print("\nReferenced Documents:")
    #for doc in retrieved_context:
        #print(doc["doc_name"])


if __name__ == "__main__":
    main()
