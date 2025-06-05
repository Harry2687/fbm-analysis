import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import modules.preprocessing as prep
    from modules.rag.models import ChatFBM
    import os
    return ChatFBM, mo, os, prep


@app.cell
def _(mo):
    mo.md(r"""## Convert json files to single txt file.""")
    return


@app.cell
def _(prep):
    prep.convert_to_txt('data/the_office', 'source_documents/the_office.txt')
    prep.convert_to_txt('data/fuck_harry_zhing', 'source_documents/fuck_harry_zhing.txt')
    return


@app.cell
def _(mo):
    mo.md(r"""## Create vector databases for different chunk sizes""")
    return


@app.cell
def _(ChatFBM, os):
    chunk_sizes = [1000, 750, 500, 250]
    chat = 'the_office'
    for size in chunk_sizes:
        model = ChatFBM()
        db_path = f'databases/{chat}_vectordb_chunksize{size}'
        overlap = int(0.1*size)
    
        if not os.path.exists(db_path):
            model.ingest(
                file_path=f'source_documents/{chat}.txt',
                db_path=db_path,
                chunk_size=size,
                chunk_overlap=overlap
            )
    return (chunk_sizes,)


@app.cell
def _(mo):
    mo.md(r"""## Check number of chunks for each chunk size""")
    return


@app.cell
def _(ChatFBM, chunk_sizes, os):
    chat = 'the_office'
    for size in chunk_sizes:
        db_path = f'databases/{chat}_vectordb_chunksize{size}'
    
        if os.path.exists(db_path):
            model = ChatFBM()
            model.load_db(db_path)
            n_chunks = len(model.vector_store.get()['documents'])
            print(f'Number of chunks in {chat} for chunk size {size}: {n_chunks}')
    return


if __name__ == "__main__":
    app.run()
