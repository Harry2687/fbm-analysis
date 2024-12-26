'''
This takes a look at how similarity search works with the vector db
'''

from langchain_chroma.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

db_file_path = 'databases/chatlog_vectordb'
vector_store = Chroma(
    persist_directory=db_file_path,
    embedding_function=OllamaEmbeddings(model='nomic-embed-text')
)

query = input('Query for Chroma db: ')
results = vector_store.similarity_search(query=query, k=5)

for i, result in enumerate(results):
    print(f'Result {i+1}:')
    print(result)
    print('\n')