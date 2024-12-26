'''
This basically just takes a look at what the chunks look like
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

file_path = 'source_documents/chatlog.txt'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
docs = TextLoader(file_path=file_path).load()
chunks = text_splitter.split_documents(docs)
chunks = filter_complex_metadata(chunks)