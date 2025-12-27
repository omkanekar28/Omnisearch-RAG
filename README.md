# Omnisearch-RAG
Multi-modal RAG system supporting PDFs, images, excel, text, and URLs. Built with LangChain and FAISS for intelligent document retrieval and QA.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/omkanekar28/RAG-Implementation.git
cd RAG-Implementation
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:

1) If no GPU is available:
```bash
pip install -r requirements-cpu.txt
```

2) If GPU is available:
```bash
pip install -r requirements-gpu.txt
```

## Usage

1. Make sure Ollama is running with the required models.
```bash
ollama serve
ollama pull qwen3:1.7b
ollama pull qwen3-vl:2b
```

2. Configure parameters in src/config/config.py.

3. Run input_ingestion.py (Converts input files into text-chunks)
```bash
python3 -m src.pipeline.input_ingestion
```

4. Run embedding_indexer.py (Converts text-chunks into embeddings + metadata)
```bash
python3 -m src.pipeline.embedding_indexer
```

5. Run rag_engine.py (Performs Context-Based-Question-Answering)
```bash
python3 -m src.pipeline.rag_engine
```

### NOTE:- Make sure you uncomment the main function in each of the above files before running them for end-to-end execution.

## Demos
Links to demo videos showcasing the functionality of the system can be found in demos.txt.