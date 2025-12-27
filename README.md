# Omnisearch-RAG

**Multi‑modal Retrieval-Augmented Generation (RAG)** system that supports PDFs, images, Excel, text, and URLs. Built with LangChain and FAISS for fast semantic retrieval and Ollama for local LLM inference.

---

## Quick summary

Omnisearch-RAG ingests heterogeneous documents, chunks and embeds them, stores vectors in FAISS, and uses a local LLM to perform context-aware question answering (QA). Designed to be modular and production-friendly — great for demonstrations, interviews, and portfolio showcases.

**Highlights**

* Multi-format ingestion: PDF, images (OCR-ready), Excel, plain text, and web pages.
* Embeddings + FAISS for scalable semantic search.
* Local LLM support via Ollama (configurable to other LLM endpoints).
* Clean, modular pipeline: ingestion → embedding → retrieval → generation.

---

## Features

* ✅ Document ingestion & chunking
* ✅ Embeddings with SentenceTransformers (configurable)
* ✅ FAISS index for fast similarity search
* ✅ Context-aware prompt construction for LLM responses
* ✅ Demo scripts and sample outputs

---

## Prerequisites

* Python 3.9+ (3.10/3.11 recommended)
* Git
* Ollama (for local LLMs) or another LLM endpoint if you adapt the project
* Optional: GPU for faster embeddings / model inference

---

## Clone

```bash
git clone https://github.com/omkanekar28/Omnisearch-RAG.git
cd Omnisearch-RAG
```

---

## Install

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate  # Windows (PowerShell)
```

Install dependencies:

```bash
# If no GPU
pip install -r requirements-cpu.txt

# If you have a supported GPU
pip install -r requirements-gpu.txt
```

---

## Configuration

All runtime parameters are in `src/config/config.py`. Configure:

* Paths for input data and output
* Embedding model and options
* FAISS index settings
* Ollama / LLM server address and model names

**Ollama example:**

```bash
# Start Ollama and download required models
ollama serve

# Question-Answering Model
ollama pull qwen3:1.7b

# Image-Processing Model
ollama pull qwen3-vl:2b
```

If you plan to use different models, make the necessary changes in config.py

---

## Recommended workflow (end-to-end)

1. **Ingest documents & chunk**

```bash
python3 -m src.pipeline.input_ingestion
```

2. **Create embeddings + build FAISS index**

```bash
python3 -m src.pipeline.embedding_indexer
```

3. **Run RAG engine: query + generate answers**

```bash
python3 -m src.pipeline.rag_engine
```

> **Note:** Uncomment the `main()` in those files if you have commented them out for development.

---


## Demos

Demo links can be found in `demos.txt`.