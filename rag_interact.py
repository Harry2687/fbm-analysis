import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from modules.rag.rag_demo import rag_demo
    return mo, rag_demo


@app.cell
def _(mo):
    mo.md("""## Test queries""")
    return


@app.cell
def _(rag_demo):
    chunk_sizes = [500]
    chat_name = 'the_office'
    queries = [
        "What games does Harry play?",
        "What topics does Harry talk about with Dhruv?",
        "What topics does Harry talk about with Mansoor?",
        "What is the overall sentiment of messages sent by Harry?",
        "Which topic has generated the most positive sentiment?",
        "Which topic has generated the most negative sentiment"
    ]

    for size in chunk_sizes:
        context_size = round(5000/size)
        db_path = f'databases/{chat_name}_vectordb_chunksize{size}'

        print(f'Chunk size {size}, with {context_size} chunks for context')
        for query in queries:
            print(f'Query: {query}')
            rag_demo(
                query=query,
                db_file_path=db_path,
                context_size=context_size,
                print_docs=False,
                model_name='llama3.1:8b'
            )
            print('\n')
    return


@app.cell
def _(rag_demo):
    rag_demo(
        query='What games does Harry play?',
        db_file_path='databases/the_office_vectordb_chunksize500',
        context_size=10,
        print_docs=False,
        model_name='llama3.1:8b'
    )
    return


if __name__ == "__main__":
    app.run()
