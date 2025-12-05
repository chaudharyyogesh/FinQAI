# ingest_data.py

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# IMPORT SETTINGS FROM CONFIG
import config

def create_vector_db():
    if not os.path.exists(config.DATA_PATH):
        os.makedirs(config.DATA_PATH)
        print(f"Created folder {config.DATA_PATH}. Put PDFs there.")
        return

    print("Loading PDF files...")
    loader = DirectoryLoader(config.DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages.")

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    print("Initializing embeddings...")
    embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    print("Creating Database...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function, 
        persist_directory=config.DB_PATH,
        collection_name=config.COLLECTION_NAME
    )
    
    print(f"Success! Database created at {config.DB_PATH}")

if __name__ == "__main__":
    create_vector_db()