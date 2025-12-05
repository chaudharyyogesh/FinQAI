from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

print("Start chatting with your financial data (type 'exit' to stop).")

while True:
    query = input("User: ")
    if query.lower() == "exit":
        break

    # Invoke the RAG chain
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    response = result["answer"]
    print(f"Assistant: {response}")

    # Update history for the next turn
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response))