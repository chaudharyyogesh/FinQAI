from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# System prompt for the final answer
qa_system_prompt = """You are a financial analyst assistant. \
Use the following pieces of retrieved context to answer the question. \
If the answer is not in the context, say that you don't know. \
Keep the answer concise and professional. Also pay attention to the dates if mentioned.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# This chain processes the documents and generates the answer
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Combine retrieval and generation
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)