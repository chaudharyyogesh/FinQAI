# chat_app_clean.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import config

print("Loading Llama-3 Model... (this may take a minute)")

# 1. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -----------------------------
# 2. Define Llama chat wrapper
# -----------------------------
def llama_chat_generate(messages):
    """
    Generate text using Llama-3 chat template.
    messages: list of LC messages (HumanMessage, AIMessage, SystemMessage)
    """
    # Convert LangChain messages to OpenAI-style dict
    converted = []
    for m in messages:
        if m.type == "system":
            converted.append({"role": "system", "content": m.content})
        elif m.type == "human":
            converted.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            converted.append({"role": "assistant", "content": m.content})

    # Apply Llama chat template
    chat_prompt = tokenizer.apply_chat_template(
        converted,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize prompt
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        repetition_penalty=config.REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# LangChain LLM wrapper
class LlamaChatWrapper:
    def __call__(self, prompt_value):
        messages = prompt_value.to_messages()
        return llama_chat_generate(messages)


llm = LlamaChatWrapper()

# -----------------------------
# 3. Connect to vector DB
# -----------------------------
print("Connecting to Vector Database...")
embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

vector_db = Chroma(
    persist_directory=config.DB_PATH,
    embedding_function=embedding_function,
    collection_name=config.COLLECTION_NAME
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# 4. Setup RAG chain
# -----------------------------
print("Setting up RAG pipeline...")

qa_system_prompt = """
You are a financial assistant.
Answer ONLY the user's question based strictly on the provided context if provided.
Kepp answers short.
Context:
<start context>
{context}
<end context>

"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    ("human", "{input}")
])

# Chain to combine retrieved documents and generate answer
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -----------------------------
# 5. Chat Loop
# -----------------------------
print("\n--- System Ready! Type 'exit' to stop ---")

while True:
    query = input("\nUser: ")
    if query.lower() == "exit":
        break

    # Invoke RAG chain
    result = rag_chain.invoke({"input": query})
    print("RESULT", result)
    response = result["answer"]

    # print(f"Assistant: {response}")

