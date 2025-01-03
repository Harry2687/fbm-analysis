from langchain_chroma.vectorstores import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
import chromadb

class ChatFBM:
    def __init__(self, model_name: str='llama3.1', context_size: int=5):
        self.model = ChatOllama(model=model_name, show_progress=True)
        self.prompt = PromptTemplate.from_template(
            '''
            [INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
            Question: {question} 
            Context: {context} 
            Answer: [/INST]
            '''
        )
        self.context_size = context_size

    def ingest(self, file_path: str, db_path: str, chunk_size: int=1024, chunk_overlap: int=100):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = TextLoader(file_path=file_path).load()
        chunks = text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        chunk_limit = chromadb.Client().get_max_batch_size()

        if len(chunks) > chunk_limit:
            def split_list(input_list, chunk_size):
                for i in range(0, len(input_list), chunk_size):
                    yield input_list[i:i + chunk_size]
            
            split_chunks = split_list(chunks, chunk_limit)

            for doc_chunk in split_chunks:
                Chroma.from_documents(
                    documents=doc_chunk,
                    embedding=OllamaEmbeddings(model='nomic-embed-text'),
                    persist_directory=db_path
                )
        else:
            Chroma.from_documents(
                documents=chunks, 
                embedding=OllamaEmbeddings(model='nomic-embed-text'),
                persist_directory=db_path
            )

    def load_db(self, db_file_path: str):
        self.vector_store = Chroma(
            persist_directory=db_file_path,
            embedding_function=OllamaEmbeddings(model='nomic-embed-text')
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                'k': self.context_size,
            },
        )

        self.chain = ({'context': self.retriever, 'question': RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def retrieve_docs(self, query: str):
        results = self.vector_store.similarity_search(query=query, k=self.context_size)

        print('Context search results:\n')
        for i, result in enumerate(results):
            print(f'Result {i+1}:')
            print(result)
            print('\n')

    def ask(self, query: str):
        if not self.chain:
            return 'Please, add a Text document first.'

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None