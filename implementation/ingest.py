import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

MODEL = "gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv(override=True)


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    total_docs = 0
    for folder in folders:
        doc_type = os.path.basename(folder)
        print(f"Loading documents from folder: {folder} (Type: {doc_type})")
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        try:
            folder_docs = loader.load()
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
            print(f"Loaded {len(folder_docs)} documents from this folder.")
            total_docs += len(folder_docs)
        except Exception as e:
            print(f"Error loading documents from {folder}: {e}")

    print(f"Total documents fetched: {total_docs}")
    return documents


def create_chunks(documents):
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splitter = SemanticChunker(embeddings)

    all_chunks = []
    total_chunks_generated = 0

    print("Processing documents individually for chunking analysis:")
    for i, doc in enumerate(documents):
        doc_source = doc.metadata.get('source', f'Unknown Source {i+1}')
        print(f"\nProcessing Document {i+1}: {doc_source}")

        try:
            # Splitting one document at a time to track chunks per document
            doc_chunks = text_splitter.split_documents([doc])
            num_doc_chunks = len(doc_chunks)
            print(f"  Generated {num_doc_chunks} chunks for this document.")

            for j, chunk in enumerate(doc_chunks):
                chunk_length = len(chunk.page_content)
                print(f"    Chunk {j+1}/{num_doc_chunks}: Length = {chunk_length} characters")
                chunk.metadata['source'] = doc_source
                chunk.metadata['doc_chunk_index'] = j 
                all_chunks.append(chunk) 
                total_chunks_generated += 1

        except Exception as e:
            print(f"  Error chunking document {doc_source}: {e}")

    print(f"Total chunks generated across all documents: {total_chunks_generated}")
    return all_chunks

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )
    print("Vector store created successfully.")
    collection = vectorstore.get()
    count = len(collection.get('ids', ))

    print(f"Successfully added {count:,} items (embeddings and documents) to the vector store.")

    return vectorstore

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")