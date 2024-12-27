# Retrieval-Augmented Generation (RAG) Pipeline

This repository provides a **RAGPipeline** class that demonstrates how to:

- Load and process PDF/documents from a specified directory,
- Split text into manageable chunks using *RecursiveCharacterTextSplitter*,
- Create embeddings using a Hugging Face model,
- Store and retrieve these vectors using **Chroma**,
- Utilize a **ChatGroq** model to answer queries in both raw text and JSON-based structured formats (including confidence scores and source references).

---

## 1. Prerequisites

1. **Python 3.9+**  
   Make sure you are running a recent version of Python (3.9 or above).

2. **git** (optional, for cloning from a Git repository).

3. **.env file (Optional)**  
   If you plan to store your `GROQ_API_KEY` securely in a `.env` file, create this file in the project root directory:
   ```bash
   GROQ_API_KEY=your_groq_api_key_goes_here
This code uses `python-dotenv` to load environment variables from `.env`.


## 2. Installation

1. Clone or Download this Repository
   ```bash
   git clone https://github.com/Sayan2908/RAG-Based-Q-A-Assistant.git
   cd RAG-Based-Q-A-Assistant

3. Create a Python Environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Linux/Mac
   venv\Scripts\activate        # On Windows

5. Install dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

7. (Optional) Premare .env
   ```bash
   GROQ_API_KEY=your_groq_api_key_goes_here
to a file named `.env` in the root directory. Make sure `.env` is in your `.gitignore`.


## 3. Structure of the Code

1. `rag_pipeline.py`
   - `RAGPipeline` class:
     - Loads and splits documents into chunks.
     - Embeds text using a Hugging Face model.
     - Stores and retrieves chunks from Chroma vector DB.
     - Interacts with the ChatGroq model to answer queries (basic or enhanced/structured).
   - `main()` function:
     - Demonstrates how to instantiate the RAGPipeline and run a few sample queries (both basic and enhanced).
3. `requirements.txt`
   - Lists all libraries and version pins used by the code (e.g., langchain, transformers, llama-index, pydantic, etc.).
5. `.env` (Optional)
   -Contains sensitive or environment-specific variables, such as GROQ_API_KEY.
7. Data Directory
   - By default, the pipeline loads documents from the directory /content/data (customizable in code).
   - The library SimpleDirectoryReader from llama-index is used to read documents (PDFs, text files, etc.).
9. `RAG-BasedQ&AAssistant.ipynb`
    - Notebook version (Can be run in google Colab)

## 4. Running of the Script

1. Place your documents (e.g., PDFs) in the directory that the code points to (default: /content/data).
2. Ensure .env is ready (if you wish to store your GROQ_API_KEY there).
3. Run:
   ```bash
   python rag_pipeline.py
You should see a prompt or queries running in the console.


## 5. Code Explanation

### 5.1 The `RAGPipeline` Class

#### **Initialization:**
```python
pipeline = RAGPipeline(
   data_dir="/content/data",
   model_name="llama3-8b-8192", 
   huggingface_embed_model="sentence-transformers/all-MiniLM-L6-v2"
)
```


1. API Key: Automatically pulled from your .env, or prompted if missing
2. ChatGroq model: Creates a ChatGroq object using the specified model_name
3. Document Loading: SimpleDirectoryReader(data_dir).load_data()
4. Splitting: Uses RecursiveCharacterTextSplitter to chunk text
5. Embedding: Embeds these chunks via the specified Hugging Face model
6. Vector Storage: Chroma.from_texts(...) to persist text embeddings in a local directory
7. Retriever: Gains quick access to relevant chunks given a user query

#### **Core Methods:**

- `basic_query(query, top_k=3)`: Retrieves the top k most relevant chunks from Chroma and constructs a basic prompt for the ChatGroq model. Returns the raw text output.
- `enhanced_query(query, top_k=3)`: Constructs a more complex prompt, instructing the ChatGroq model to return a JSON-based answer with confidence scores and source references. The pipeline then parses the JSON to produce a structured RAGResponse object (fields: answer, source_documents, confidence_score).

#### **Utility Methods:**

- _split_documents(docs): Splits documents into multiple chunks
- _store_in_chroma(text_chunks, embeddings): Stores chunk embeddings in Chroma
- _parse_rag_response(content): Attempts to parse or regex-extract JSON from the model's output
- _extract_json(text): Helper for raw JSON extraction if direct parsing fails

### 5.2 `main()` Function
*Demonstrates how to:*
- Instantiate the pipeline
- Run queries (both basic and enhanced)
- Print results
