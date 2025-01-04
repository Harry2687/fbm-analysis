from .models import ChatFBM

def rag_demo(query: str, db_file_path: str, model_name: str='llama3.1', context_size: int=5, print_docs: bool=True):
    chat = ChatFBM(model_name, context_size)
    chat.load_db(db_file_path)
    if print_docs:
        chat.retrieve_docs(query)
    print('LLM response:')
    print(chat.ask(query))