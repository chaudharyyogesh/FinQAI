import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# IMPORT SETTINGS FROM CONFIG
import config

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="JPM Financial Analyst", layout="wide")
st.title("📊 JPM Quarterly Results AI Analyst")

# --- 1. LOAD MODEL (CACHED) ---
# We use @st.cache_resource to load the model only ONCE, not on every user interaction.
@st.cache_resource
def load_llm():
    print("Loading Model... (this may take a minute)")

    model_id = config.MODEL_PATH

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" # or device="cuda"
    )

    # 2. Define Terminators
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 3. Create the Pipeline
    # Pass all generation parameters here so LangChain picks them up automatically
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        repetition_penalty=config.REPETITION_PENALTY,
        top_p=config.TOP_P,
        return_full_text=False,
        do_sample=True,        # Enable sampling
        eos_token_id=terminators, # Pass terminators here
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm

# --- 2. LOAD DATABASE & CHAIN (CACHED) ---
@st.cache_resource
def load_rag_chain(_llm):
    print("Connecting to Database...")
    embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    vector_db = Chroma(
        persist_directory=config.DB_PATH,
        embedding_function=embedding_function,
        collection_name=config.COLLECTION_NAME
    )

    target_file = "financial_pdfs/jpmorgan_3q24_pr.pdf"
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 6,
        "filter": {"source": target_file}}
    )

    print("Loading Reranker Model...")
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

    # 3. Create the Compression Retriever
    # This wrapper handles the "Retrieve k -> Score -> Pick Top n" logic automatically
    compressor = CrossEncoderReranker(model=model, top_n=5)

    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever
    )

    # Prompt Template
    # system_prompt = (
    #     "You are a financial analyst assistant. "
    #     "Use the context to answer the user question."
    #     "Pay specific attention to dates while answering the user questions. "
    #     "If the answer is not in the context, say 'I cannot find that information in the context provided'. "
    #     "Do not hallucinate numbers. "
    #     "\n\n"
    #     "Context:\n"
    #     "{context}"
    # )

    llama_prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a financial analyst assistant. Use the following context to answer the question.
        Pay specific attention to dates while answering the user questions.
        If the answer is not in the context, say "I cannot find that information".
        Do not hallucinate numbers.

        Context:
        {context}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_prompt),
    #         ("human", "{input}"),
    #     ]
    # )
    prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=llama_prompt_template
    )   

    question_answer_chain = create_stuff_documents_chain(_llm, prompt)
    rag_chain = create_retrieval_chain(rerank_retriever, question_answer_chain)
    return rag_chain

# --- INITIALIZE RESOURCES ---
try:
    with st.spinner("Initializing Model & Database..."):
        llm = load_llm()
        rag_chain = load_rag_chain(llm)
        st.success("System Ready!")
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# --- CHAT HISTORY (SESSION STATE) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- MAIN CHAT INTERFACE ---
if query := st.chat_input("Ask a question about the quarterly reports (e.g., 'What was the Net Income in Q3 2024?'):"):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            response_dict = rag_chain.invoke({"input": query})
            answer = response_dict["answer"]
            
            # Display Answer
            st.markdown(answer)
            
            # Display Sources in an Expander (Collapsible)
            with st.expander("View Source Documents"):
                for i, doc in enumerate(response_dict["context"]):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    st.markdown(f"**Source {i+1}:** {source} (Page {page})")
                    st.caption(doc.page_content[:300] + "...") # Preview first 300 chars
                    st.divider()

    # 3. Save Assistant Message to History
    st.session_state.messages.append({"role": "assistant", "content": answer})