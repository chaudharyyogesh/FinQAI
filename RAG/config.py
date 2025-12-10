# config.py

import os

# --- Paths ---
# The folder where you put your PDFs
DATA_PATH = "./financial_pdfs"

# The folder where the Chroma DB will be created
DB_PATH = "./chroma_langchain_db"

# The path to your local fine-tuned Llama model
# UPDATE THIS to the actual name of your model folder!
MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct" 

# --- Model Settings ---
# MUST be the same for ingestion and chat. 
# If you change this, you must delete 'chroma_db_data' and re-run ingest_data.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Database Settings ---
COLLECTION_NAME = "financial_reports"

# --- Generation Settings ---
# Adjust these to change how the Llama model behaves
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1  # Low temperature = more factual
REPETITION_PENALTY = 1.1
TOP_P = 0.9
CONTEXT_WINDOW = 4096 # Depends on your specific Llama version