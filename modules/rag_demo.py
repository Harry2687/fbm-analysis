from modules.models import ChatFBM

def rag_demo(query: str, db_file_path: str, model_name: str='llama3.1', context_size: int=5):
    chat = ChatFBM(model_name, context_size)
    chat.load_db(db_file_path)
    chat.retrieve_docs(query)
    print('LLM response:')
    print(chat.ask(query))