from langchain_chroma.vectorstores import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

class ChatFbMessenger:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama3.1", show_progress=True)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            [INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
            Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, file_path: str, db_path: str):
        docs = TextLoader(file_path=file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=db_path
        )
        self.retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ingest_db(self, db_file_path: str):
        vector_store = Chroma(
            persist_directory=db_file_path,
            embedding_function=OllamaEmbeddings(model='nomic-embed-text'),
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def retrieve_docs(self, query: str):
        return self.retriever.invoke(query)

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a Text document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

theoffice = ChatFbMessenger()
theoffice.ingest_db('databases/chatlog_vectordb')
query = input('What would you like to know about The Office?\n')
print(theoffice.ask(query))