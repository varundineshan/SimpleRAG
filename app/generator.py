# generator.py

from openai import AzureOpenAI
from dotenv import load_dotenv
from .config import GENERATIVE_MODEL
import os
load_dotenv()
# Create a client for embedding calls (using a fixed API version)
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")


def generate_answer(query, context_docs):
    """
    Generate an answer using the retrieved context and the user query.

    Args:
        query (str): The user query.
        context_docs (list): List of dicts with keys "doc_text" and "doc_name".

    Returns:
        str: The generated answer.
    """
    # Create a new client for the chat completions (with a preview API version)
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-08-01-preview"
    )

    # Combine context documents into a single string with document names as headers
    context_str = "\n\n".join(
        [f"Document: {doc['doc_name']}\n{doc['doc_text']}" for doc in context_docs]
    )

    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                            "You are a knowledgeable assistant. Based on the following context, "
                            "answer the question. Context: " + context_str
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                }
            ]
        }
    ]

    messages = chat_prompt
    # Generate the completion
    completion = client.chat.completions.create(
        model=GENERATIVE_MODEL,
        messages=messages,
        temperature=0.7,
        top_p=0.95
    )

    # Extract the answer from the first choice
    completion_choices = completion.__getattribute__('choices')
    first_choice = completion_choices[0]
    answer = first_choice.message.content
    return answer
