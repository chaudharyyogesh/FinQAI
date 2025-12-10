import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 1. Import your RAG Chain (Assuming you refactored chat_app.py to be importable)
# If you haven't refactored, you might need to copy your chain setup code here.
# For this example, let's assume we can import 'rag_chain' from a module.
# from app import rag_chain 

# --- MOCK SETUP (Replace this with your actual rag_chain import) ---
# This block simulates your pipeline for the sake of this script working immediately.
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import config

print("Initializing Pipeline for Evaluation...")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATH, device_map="auto", dtype=torch.float16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, return_full_text=False)
llm = HuggingFacePipeline(pipeline=pipe)
embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
vector_db = Chroma(persist_directory=config.DB_PATH, embedding_function=embedding_function, collection_name=config.COLLECTION_NAME)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

system_prompt = "You are a financial analyst. Answer based on context. Context: {context}"
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# ------------------------------------------------------------------


# 2. Define Your Test Data (Ground Truth)
# You need a few questions and the ACTUAL correct answers (Ground Truth) to measure against.
data_samples = {
    'question': [
        "What was the Net Income for Q3 2024?", 
        "Did revenue increase or decrease in Q3 2024 compared to Q3 2023?",
        "What are the risks mentioned regarding interest rates?"
    ],
    'ground_truth': [
        "Net income was $12.9 billion.", 
        "Revenue increased by 21% to $42.7 billion.",
        "The report mentions risks related to higher-for-longer interest rates and their impact on deposit costs."
    ]
}

# 3. Generate Responses from Your RAG Pipeline
print("Running generation on test set...")
answers = []
contexts = []

for query in data_samples['question']:
    print(f"Processing: {query}")
    response = rag_chain.invoke({"input": query})
    
    # Store the generated answer
    answers.append(response["answer"])
    
    # Store the retrieved text chunks (content only)
    retrieved_docs = [doc.page_content for doc in response["context"]]
    contexts.append(retrieved_docs)

# 4. Prepare Dataset for Ragas
dataset_dict = {
    "question": data_samples['question'],
    "answer": answers,
    "contexts": contexts,
    "ground_truth": data_samples['ground_truth']
}
ds = Dataset.from_dict(dataset_dict)

# 5. Run Evaluation
# make sure you have os.environ["OPENAI_API_KEY"] set, or configure a local critic.
print("Evaluating metrics with Ragas (using OpenAI GPT-4 as Judge)...")

results = evaluate(
    ds,
    metrics=[
        faithfulness,      # Did the model make things up?
        answer_relevancy,  # Did it actually answer the question?
        context_precision, # Was the relevant info in the top chunks?
        context_recall,    # Did we retrieve the info needed to answer the ground truth?
    ],
)

# 6. Show Results
print("\n--- Evaluation Results ---")
print(results)

# Save detailed results to CSV for your report
df = results.to_pandas()
df.to_csv("rag_evaluation_results.csv", index=False)
print("Detailed results saved to 'rag_evaluation_results.csv'")