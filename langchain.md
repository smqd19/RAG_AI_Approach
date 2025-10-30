# LangChain 1.0 Migration Guide

## Overview

LangChain 1.0 represents a major simplification of the framework. The key change for RAG applications: **helper chains like `create_retrieval_chain` and `create_stuff_documents_chain` are GONE**.

The new approach is simple Python code: retrieve documents, format context, call the LLM. No chains, no LCEL complexity - just straightforward function calls.

**Key Philosophy**: Write simple, explicit Python code. LangChain 1.0 provides the building blocks (retrievers, LLMs, embeddings) but leaves the orchestration to you.

---

## What Changed in 1.0

### What's Gone

❌ `from langchain.chains import create_retrieval_chain` - **REMOVED**
❌ `from langchain.chains.combine_documents import create_stuff_documents_chain` - **REMOVED**
❌ `from langchain.chains import RetrievalQA` - **REMOVED**
❌ LCEL (LangChain Expression Language) pipe syntax - **Not commonly used in practice**

### What's Here

✅ Simple method calls: `retriever.invoke()`, `llm.invoke()`
✅ Provider-specific packages: `langchain-openai`, `langchain-chroma`, etc.
✅ Text splitters: `langchain-text-splitters`
✅ Document loaders: `langchain-community`
✅ Plain Python code

---

## Package Structure in 1.0

### Core Packages (Already in Your Project)

```toml
langchain>=1.0.2              # Minimal - mostly empty in 1.0
langchain-core>=1.0.0         # Base classes (Document, messages, etc.)
langchain-text-splitters>=1.0.0   # Text chunking
langchain-community>=0.4      # Document loaders

# Provider packages
langchain-openai>=1.0.1       # OpenAI models and embeddings
langchain-huggingface>=1.0.0  # HuggingFace embeddings
langchain-chroma>=1.0.0       # Chroma vector store
```

**Note**: The `langchain` package is essentially empty in 1.0. Everything is in provider-specific packages.

### Import Patterns

```python
# Document loading and chunking
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store
from langchain_chroma import Chroma

# LLM
from langchain_openai import ChatOpenAI

# Messages and documents
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
```

---

## Three Approaches to RAG in LangChain 1.0

LangChain 1.0 offers three ways to build RAG applications. Here's what you need to know:

### 1. LangGraph (Official, But Heavy)

**What it is:** LangChain's official tutorial recommends LangGraph - a state-based orchestration framework.

**Pattern:**
```python
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "What is Task Decomposition?"})
```

**When to use:**
- ✅ Complex agentic workflows with branching, cycles, multi-step reasoning
- ✅ Need for human-in-the-loop, persistence, durable execution
- ✅ Building sophisticated multi-agent systems

**When NOT to use:**
- ❌ Simple RAG (retrieve + generate) - it's overkill
- ❌ You want clear, debuggable code
- ❌ You're building a straightforward Q&A system

**Verdict for your use case:** Too heavy. You don't need state graphs for simple RAG.

---

### 2. LCEL (Exists, But Not Popular)

**What it is:** LangChain Expression Language - the pipe syntax (`|`) for composing chains.

**Pattern:**
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Single-invoke chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Use it
answer = rag_chain.invoke("Who is Avery?")
```

**When to use:**
- ✅ You want a single `.invoke()` call
- ✅ Simple linear chains (prompt → llm → parser)
- ✅ You like the functional pipe syntax

**When NOT to use:**
- ❌ Complex workflows (LangChain docs say "use LangGraph instead")
- ❌ You find the pipe syntax harder to debug
- ❌ You prefer explicit control flow

**Verdict:** LCEL still exists in 1.0 and technically works, but it hasn't gained wide adoption in practice. LangChain tried to promote it heavily 1-2 years ago, but developers prefer simpler approaches. The docs even say "if you're building a complex chain... use LangGraph instead," which leaves LCEL in a narrow middle ground.

---

### 3. Simple Python (Practical, Recommended)

**What it is:** Manual orchestration with explicit function calls - no magic.

**Pattern:**
```python
def answer_question(question: str) -> tuple[str, list]:
    # 1. Retrieve
    docs = retriever.invoke(question)

    # 2. Format
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3. Build prompt
    system_prompt = f"Answer based on context:\n{context}"

    # 4. Generate
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content, docs
```

**When to use:**
- ✅ Simple RAG applications (retrieve → generate)
- ✅ You want clear, debuggable code
- ✅ You want explicit control over each step
- ✅ You're building straightforward Q&A systems

**When NOT to use:**
- ❌ You really need LangGraph's state management
- ❌ You want a single-line `.invoke()` call (though this is debatable)

**Verdict for your use case:** **This is the recommended approach.** It's simple, explicit, easy to debug, and doesn't add unnecessary abstraction. This is what practical developers are using for basic RAG.

---

## Our Recommendation: Simple Python

For your simple RAG use case, **use the manual approach (#3)**. Here's why:

1. **LangGraph is overkill** - You don't have complex branching, cycles, or multi-agent workflows
2. **LCEL didn't gain traction** - It adds abstraction without clear benefits for simple cases
3. **Simple Python is clearest** - Easy to understand, debug, and customize

If you later need complex orchestration, you can always upgrade to LangGraph. But start simple.

---

## The Simple RAG Pattern

### Old Way (Pre-1.0) - NO LONGER WORKS

```python
# ❌ This doesn't work in 1.0
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
result = rag_chain.invoke({"input": "Who is Avery?"})
```

### New Way (1.0) - Simple Python

```python
# ✅ This is the LangChain 1.0 way
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def answer_question(question: str) -> tuple[str, list]:
    """Simple RAG: retrieve, format, generate"""

    # 1. Retrieve relevant documents
    docs = retriever.invoke(question)

    # 2. Format documents into context string
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3. Build prompt with context
    system_prompt = f"""Answer based on the context below.
If you don't know, say so.

Context:
{context}"""

    # 4. Call LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content, docs
```

**That's it!** No chains, no LCEL, no magic - just:
1. `retriever.invoke()` to get documents
2. Join documents into a context string
3. `llm.invoke()` with system + human messages
4. Return the answer

---

## Migration Strategy

### 1. Document Loading (No Change)

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
import glob

def load_documents():
    """Load documents from knowledge base"""
    folders = glob.glob("knowledge-base/*")
    documents = []

    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()

        # Add metadata
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents
```

**Best Practices:**
- Always specify `encoding="utf-8"` for TextLoader
- Use `glob` parameter to filter files
- Add custom metadata to documents

---

### 2. Text Chunking (No Change)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)
```

**How it works:**
- Tries to split by: `\n\n` (paragraphs), then `\n` (lines), then spaces, then characters
- Keeps related content together when possible
- `chunk_overlap` ensures context continuity

**Common chunk sizes:**
- Small context models: 500-800
- Medium context: 1000-1500
- Large context: 2000-4000

---

### 3. Embeddings (Package Change Only)

```python
# OpenAI embeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
# or specify model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# HuggingFace embeddings (local, no API key)
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

**Migration note:** Change `from langchain.embeddings` to provider-specific imports.

---

### 4. Vector Store - Chroma (Package Change)

```python
from langchain_chroma import Chroma
import os

def create_vector_store(chunks, embeddings, db_name="vector_db"):
    """Create and persist vector store"""

    # Delete existing collection
    if os.path.exists(db_name):
        Chroma(
            persist_directory=db_name,
            embedding_function=embeddings
        ).delete_collection()

    # Create new vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_name
    )

    return vectorstore

def load_vector_store(embeddings, db_name="vector_db"):
    """Load existing vector store"""
    return Chroma(
        persist_directory=db_name,
        embedding_function=embeddings
    )
```

**Query patterns:**

```python
# Direct similarity search
results = vectorstore.similarity_search("question", k=5)

# Similarity search with scores
results = vectorstore.similarity_search_with_score("question", k=5)

# As retriever (returns Runnable)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
docs = retriever.invoke("question")
```

---

### 5. RAG Implementation (Major Change)

#### Complete Example: ingest.py

```python
import os
import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuration
DB_NAME = "vector_db"
KNOWLEDGE_BASE = "knowledge-base"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_documents():
    folders = glob.glob(f"{KNOWLEDGE_BASE}/*")
    documents = []

    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()

        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    embeddings = get_embeddings()

    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )

    print(f"Created vector store with {vectorstore._collection.count()} chunks")
    return vectorstore

if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    create_vector_store(chunks)
    print("Ingestion complete")
```

#### Complete Example: answer.py

```python
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuration
MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"
K_CHUNKS = 10

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant for InsureLLM.
Answer the question based on the provided context.
If you don't know the answer, say so.

Context:
{context}"""

# Initialize components
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=get_embeddings()
)

retriever = vectorstore.as_retriever(search_kwargs={"k": K_CHUNKS})
llm = ChatOpenAI(model=MODEL, temperature=0)

def fetch_context(question: str) -> list[Document]:
    """Retrieve relevant context documents"""
    return retriever.invoke(question)

async def answer_question(question: str) -> tuple[str, list[Document]]:
    """Answer a question using RAG - simple Python approach"""

    # 1. Retrieve documents
    docs = retriever.invoke(question)

    # 2. Format context
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3. Build prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    # 4. Call LLM (async)
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content, docs

# Sync version
def answer_question_sync(question: str) -> tuple[str, list[Document]]:
    """Synchronous version"""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content, docs
```

---

## Async Support

LangChain 1.0 has built-in async support - just use `ainvoke()` instead of `invoke()`:

```python
# Sync
docs = retriever.invoke(question)
response = llm.invoke(messages)

# Async
docs = await retriever.ainvoke(question)
response = await llm.ainvoke(messages)
```

---

## Streaming Support

For streaming responses:

```python
def stream_answer(question: str):
    """Stream answer token by token"""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)

# Async streaming
async def astream_answer(question: str):
    """Async stream answer"""
    docs = await retriever.ainvoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    async for chunk in llm.astream(messages):
        print(chunk.content, end="", flush=True)
```

---

## Common Patterns

### Return Both Answer and Context

```python
def answer_with_sources(question: str) -> dict:
    """Return answer with source documents"""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return {
        "answer": response.content,
        "sources": docs,
        "context": context
    }
```

### Filtered Retrieval

```python
def answer_with_filter(question: str, doc_type: str) -> tuple[str, list]:
    """Retrieve only from specific document type"""
    docs = vectorstore.similarity_search(
        question,
        k=10,
        filter={"doc_type": doc_type}
    )

    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content, docs
```

### Multi-turn Conversation

```python
def answer_with_history(question: str, chat_history: list) -> str:
    """Answer with conversation history"""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context))
    ]

    # Add chat history
    messages.extend(chat_history)

    # Add current question
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content
```

---

## What to Use (and What to Avoid)

### ✅ Recommended for Simple RAG

**Simple Python functions** - The practical choice
- Easy to understand and debug
- Direct method calls: `retriever.invoke()`, `llm.invoke()`
- Plain control flow: if/else, loops, function calls
- **Use this for your RAG application**

### ⚠️ Available But Not Recommended

**LCEL (LangChain Expression Language)** - Pipe syntax like `retriever | format_docs | llm`
- Technically works in 1.0 and still supported
- Provides single `.invoke()` call if that's important to you
- But: didn't gain wide adoption in practice
- But: harder to debug than explicit code
- **Skip this unless you specifically want the pipe syntax**

**LangGraph** - State-based orchestration
- Official LangChain recommendation for RAG
- Powerful for complex agentic workflows with branching, cycles, multi-agent systems
- But: overkill for simple retrieve → generate RAG
- But: adds significant complexity
- **Only use if you need advanced orchestration features**

### ❌ Truly Gone

**langchain-classic** - Legacy chains
- Only needed if maintaining old pre-1.0 code
- Not needed for new implementations
- **Don't use for new projects**

### Bottom Line

For simple RAG: **Use simple Python functions**. The other options exist but add complexity without clear benefits for your use case. You can always upgrade to LangGraph later if your needs become more complex.

---

## Key Migration Steps for Your Code

### Your Current Code Uses:

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
result = rag_chain.invoke({"input": query})
```

### Migrate To:

```python
def answer_question(question: str) -> tuple[str, list]:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = f"""Answer based on context:
{context}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content, docs
```

### Changes Required:

1. Remove chain imports (they don't exist in 1.0)
2. Replace chain.invoke() with simple function
3. Keep your current functions `fetch_context()` and `answer_question()` - just change the implementation

---

## Testing Your Setup

Create a test file:

```python
# test_langchain_setup.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage

print("✅ All imports successful!")
print("✅ Document loaders:", TextLoader)
print("✅ Text splitters:", RecursiveCharacterTextSplitter)
print("✅ Embeddings:", HuggingFaceEmbeddings)
print("✅ Vector store:", Chroma)
print("✅ LLM:", ChatOpenAI)
print("✅ Messages:", SystemMessage, HumanMessage)
```

Run: `uv run test_langchain_setup.py`

---

## Summary

### LangChain 1.0 in a Nutshell

1. **No old chains** - `create_retrieval_chain`, `create_stuff_documents_chain` are gone
2. **Three approaches available** - LangGraph (official, heavy), LCEL (exists, not popular), Simple Python (recommended)
3. **Simple Python wins** - Write straightforward functions with `invoke()`
4. **Provider packages** - Everything is in `langchain-<provider>` packages
5. **Same building blocks** - Document loaders, text splitters, embeddings still work

### For Your RAG App

✅ **Keep using:**
- `DirectoryLoader`, `TextLoader` from `langchain_community`
- `RecursiveCharacterTextSplitter` from `langchain_text_splitters`
- `Chroma` from `langchain_chroma`
- `HuggingFaceEmbeddings` from `langchain_huggingface`
- `ChatOpenAI` from `langchain_openai`

✅ **Change to:**
- Simple Python functions instead of chains
- `retriever.invoke()` + `llm.invoke()` instead of chain helpers
- Direct control flow instead of LCEL pipes

### Key Takeaway

**LangChain 1.0 gives you choices, but simple is best for basic RAG.**

- The official tutorial recommends LangGraph, but it's overkill for simple retrieve → generate workflows
- LCEL (pipe syntax) still exists but hasn't gained wide adoption
- **Simple Python functions are the practical choice**: retrieve documents, format context, call LLM

No magic, no complexity, just simple, debuggable code.

### The Three Approaches at a Glance

| Approach | Status | Best For | Use for Simple RAG? |
|----------|--------|----------|---------------------|
| **Simple Python** | Recommended | Simple RAG, Q&A systems | ✅ **Yes** - Start here |
| **LCEL** | Available | Single-invoke chains | ⚠️ Optional - if you really want pipes |
| **LangGraph** | Official | Complex agents, workflows | ❌ No - too heavy |

Start with Simple Python. Upgrade to LangGraph only when you actually need complex orchestration.
