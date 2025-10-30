from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Any, Optional 

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# RETRIEVAL_K = 3
MAX_CONTEXT_LENGTH = 5000

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

Context:
{context}
"""

COMPRESSION_PROMPT= """
You are a highly intelligent text analysis assistant. You will be given a user's query and a set of context chunks retrieved by a 
search system, ordered by relevance.
Your task is to analyze all the context chunks and extract *only* the information that is directly relevant to answering the user's query.
Synthesize this relevant information into a single, dense, and coherent block of text.
The *entire* output text you generate MUST be less than 5000 characters.
If no relevant information is found in any of the chunks, respond ONLY with the exact string: "NO_RELEVANT_INFORMATION"

Context Chunks:
{context_chunks}

User Query: {question}

Relevant Information (under 5000 chars):
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# retriever = vectorstore.as_retriever()

# Stage 1 - Bi-Encoder
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

# Stage 2 - Cross-Encoder
model_name = "BAAI/bge-reranker-base"
model_kwargs = {'device': 'cpu'}  
model = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
compressor = CrossEncoderReranker(model=model, top_n=3)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

llm = ChatOpenAI(temperature=0, model_name=MODEL)

compression_prompt = ChatPromptTemplate.from_template(COMPRESSION_PROMPT)
compression_chain = compression_prompt | llm | StrOutputParser()

def fetch_context(question: str) -> list:
    """
    Retrieve relevant context documents for a question.
    (This now uses the two-stage re-ranking retriever)
    """
    # return retriever.invoke(question, k=RETRIEVAL_K)
    reranked_docs = retriever.invoke(question)
    print(f"Retrieved and re-ranked {len(reranked_docs)} final documents.")
    return reranked_docs

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    docs = fetch_context(question)
    final_docs_used = docs

    context_for_compression = "\n\n---\n\n".join(
        getattr(doc, "page_content", "") for doc in docs
    )

    compressed_context = compression_chain.invoke({
        "context_chunks": context_for_compression,
        "question": question
    })

    final_context_length = len(compressed_context)
    if final_context_length > MAX_CONTEXT_LENGTH:
        print(f"WARNING: LLM Compressor output exceeded limit! (Length: {final_context_length}). Truncating...")
        final_context = compressed_context[:MAX_CONTEXT_LENGTH]
    else:
        final_context = compressed_context
        
    print(f"--- FINAL COMPRESSED CONTEXT BUILT (Length: {final_context_length} chars) ---")

    system_prompt = SYSTEM_PROMPT.format(context=final_context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, final_docs_used