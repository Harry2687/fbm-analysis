import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from langchain_ollama.chat_models import ChatOllama
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain_chroma.vectorstores import Chroma
    from langchain_ollama.embeddings import OllamaEmbeddings
    from langchain.schema.output_parser import StrOutputParser
    return ChatOllama, Chroma, OllamaEmbeddings


@app.cell
def _(ChatOllama, Chroma, OllamaEmbeddings):
    llm = ChatOllama(model='llama3.1:8b')

    vector_store = Chroma(
        persist_directory='databases/the_office_vectordb_chunksize500',
        embedding_function=OllamaEmbeddings(model='nomic-embed-text')
    )
    return llm, vector_store


@app.cell
def _():
    # retriever = vector_store.as_retriever(search_kwargs={'k': 5})

    # rag_chain = (
    #     {
    #         'context': retriever,
    #         'question': RunnablePassthrough()
    #     }
    #     | prompt
    #     | model
    #     | StrOutputParser()
    # )

    # for token in rag_chain.stream('tell me about harry zhong'):
    #     print(token, end='')
    return


@app.cell
def _():
    from langgraph.graph import MessagesState, StateGraph

    graph_builder = StateGraph(MessagesState)
    return MessagesState, graph_builder


@app.cell
def _(vector_store):
    from langchain_core.tools import tool


    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=5)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    return (retrieve,)


@app.cell
def _(MessagesState, llm, retrieve):
    from langchain_core.messages import SystemMessage
    from langgraph.prebuilt import ToolNode


    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}


    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])


    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}
    return generate, query_or_respond, tools


@app.cell
def _(generate, graph_builder, query_or_respond, tools):
    from langgraph.graph import END
    from langgraph.prebuilt import tools_condition

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    # graph = graph_builder.compile()
    return


@app.cell
def _(graph_builder):
    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    # Specify an ID for the thread
    config = {"configurable": {"thread_id": "abc123"}}
    return config, graph


@app.cell
def _(config, graph):
    input_message = "Who typically talks about it?"

    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        step["messages"][-1].pretty_print()
    return


if __name__ == "__main__":
    app.run()
