"""
Prompt templates for the RAG system.
"""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are a helpful domain assistant. "
    "Answer using ONLY the provided context. "
    "If the answer is not in the context, respond with: 'I don't have enough information from the provided documents.' "
    "Be concise and cite the sources list by filename when relevant."
)

# Chat prompt expecting variables: context, question
RAG_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human",
         "Context:\n{context}\n\n"
         "Question: {question}\n\n"
         "Answer (keep factual, grounded, and concise):")
    ]
)
