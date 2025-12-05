# chat_app.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# IMPORT SETTINGS FROM CONFIG
import config

print("Loading Model... (this may take a minute)")

# 1. Load Local Model using paths from config
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True 
)

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=config.MAX_NEW_TOKENS,
    temperature=config.TEMPERATURE,
    repetition_penalty=config.REPETITION_PENALTY,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# 2. Connect to DB using paths from config
print("Connecting to Database...")
embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

vector_db = Chroma(
    persist_directory=config.DB_PATH,
    embedding_function=embedding_function,
    collection_name=config.COLLECTION_NAME
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Setup Chains (Same as before)
print("Setting up RAG pipeline...")

# Contextualize question prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer prompt
qa_system_prompt = """You are a financial analyst assistant. \
Use the following pieces of retrieved context to answer the question. \
If the answer is not in the context, say that you don't know. \
Keep the answer concise.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 4. Chat Loop
chat_history = []
print("\n--- System Ready! Type 'exit' to stop ---")

while True:
    query = input("\nUser: ")
    if query.lower() == "exit":
        break

    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    response = result["answer"]
    
    print(f"Assistant: {response}")

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response))