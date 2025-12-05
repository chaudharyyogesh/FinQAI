from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the embedding model used for your DB
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to your local Chroma DB
vector_db = Chroma(
    persist_directory="./chroma_db_data", # Path where your chroma data is saved
    embedding_function=embedding_function,
    collection_name="financial_reports"
)

retriever = vector_db.as_retriever(search_kwargs={"k": 5})