# chat_app.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# IMPORT SETTINGS FROM CONFIG
import config

print("Loading Model... (this may take a minute)")

# 1. Load Local Model using paths from config
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_PATH,
    device_map="auto",
    dtype=torch.float16, 
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=config.MAX_NEW_TOKENS,
    temperature=config.TEMPERATURE,
    repetition_penalty=config.REPETITION_PENALTY,
    top_p=config.TOP_P,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=terminators,
    return_full_text=False
)

llm = HuggingFacePipeline(
    pipeline=text_generation_pipeline
    )

# 2. Connect to DB using paths from config
print("Connecting to Database...")
embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

vector_db = Chroma(
    persist_directory=config.DB_PATH,
    embedding_function=embedding_function,
    collection_name=config.COLLECTION_NAME
)

retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.7})

# 3. Setup Chains (Same as before)
print("Setting up RAG pipeline...")

# Answer prompt
# qa_system_prompt = """You are a financial QA. \
# Answer the user's question in brief based strictly on the context if provided below. \
# If the question cannot be answered based on the context provided just respond with 'Cannot Answer. Not Found.' and stop. \

# Context:
# {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
# {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# """

# qa_prompt = ChatPromptTemplate.from_template(qa_system_prompt)

system_prompt = (
    "You are a financial analyst assistant. "
    "Use the following pieces of retrieved context to answer the question. "
    "Pay specific attention to dates."
    "If the answer is not in the context, say 'I cannot find that information in the documents'. "
    "Do not hallucinate numbers. "
    "\n\n"
    "Context:\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 4. Chat Loop
chat_history = []
print("\n--- System Ready! Type 'exit' to stop ---")

while True:
    query = input("\nUser: ")
    if query.lower() == "exit":
        break

    response = rag_chain.invoke({"input": query})
    print("RESULT:", response)
    answer = response["answer"]

    print(f"Assistant: {answer}")
    print("\n--- Sources Used ---")

    for doc in response["context"]:
        # Show Page number and Document Title
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        print(f"- {source} (Page {page})")