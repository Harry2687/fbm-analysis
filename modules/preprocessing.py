import os
import json
import re
import pandas as pd
import gensim as gs

def ms_import_data(directory: str) -> pd.DataFrame:
    data_file_names = os.listdir(directory)
    data_files = [os.path.join(directory, data_file_name) for data_file_name in data_file_names]

    messenger_data = pd.DataFrame()

    for data_file in data_files:
        with open(data_file) as file:
            data = json.load(file)
        json_data = pd.json_normalize(data['messages'])
        messenger_data = pd.concat([messenger_data, json_data])

    messenger_data = (
        messenger_data[[
            'sender_name',
            'timestamp_ms',
            'content'
        ]]
        .dropna()
        .sort_values('timestamp_ms', ascending=True)
    )

    messenger_data['timestamp'] = pd.to_datetime(messenger_data['timestamp_ms'], unit='ms')

    messenger_data = messenger_data[[
        'sender_name',
        'timestamp',
        'content'
    ]]

    return messenger_data

def remove_custom_stopwords(document: str, stopwords: list) -> str:
    for word in stopwords:
        pattern = r'\b'+word+r'\b'
        document = re.sub(pattern, '', document).replace('  ', ' ')
    
    return document

def lda_preprocess(dataframe: pd.DataFrame, content_col: str, cleaned_col: str, rm_stopwords: bool=True) -> pd.DataFrame:
    dataframe[cleaned_col] = (
        dataframe[content_col]
        .str.lower()
        .str.strip()
        .str.replace('[^a-z\\s]', '', regex=True) # remove anything that isn't text or spaces
        .str.replace('\\s{2,}', ' ', regex=True) # replace 2+ spaces with single space
    )

    chat_actions = [
        'reacted to your message'
    ]

    # remove chat actions which aren't part of the conversation
    dataframe = dataframe[
        ~dataframe[cleaned_col]
        .str.contains(
            '|'.join(chat_actions)
        )
    ]

    # stopwords which are not included in gensim
    custom_stopwords = [
        'u', 
        'lmao', 
        'lol', 
        'ur', 
        'like', 
        'yea', 
        'thats', 
        'nah', 
        'im', 
        'yeh', 
        'dont',
        'yeah', 
        'gonna', 
        'didnt',
        'idk',
        'got',
        'r',
        'sure',
        'come',
        'stuff'
        'k'
    ]

    if rm_stopwords:
        # remove gensim and custom stopwords
        dataframe[cleaned_col] = (
            dataframe[cleaned_col]
            .apply(gs.parsing.preprocessing.remove_stopwords)
            .apply(remove_custom_stopwords, args=(custom_stopwords,))
            .str.strip()
            .str.replace('\\s{2,}', ' ', regex=True)
        )

    return dataframe

def sender_wordcount(dataframe: pd.DataFrame, search_word: str) -> pd.DataFrame:
    dataframe['clean_content_splitlist'] = (
        dataframe['clean_content']
        .str.split(' ')
    )

    chat_data_wcount = (
        dataframe[['sender_name', 'clean_content_splitlist']]
        .explode('clean_content_splitlist')
        .query('clean_content_splitlist == @search_word')
        .value_counts()
        .reset_index()
        .drop('clean_content_splitlist', axis=1)
        .sort_values('count', ascending=False)
    )

    return chat_data_wcount

def lda_getdocs(dataframe: pd.DataFrame, content_col: str, ts_col: str, conv_cutoff: int=600):
    # calculate difference between each message
    dataframe['time_diff'] = (
        dataframe[ts_col]
        .diff()
        .fillna(pd.Timedelta(seconds=0))
    )
    dataframe['time_diff'] = (
        dataframe['time_diff']
        .dt.total_seconds()
    )

    # group dataframe into different conversations based on cutoff value
    dataframe['new_conv'] = dataframe['time_diff'] > conv_cutoff
    dataframe['conv_num'] = 'Conv ' + (
        dataframe['new_conv']
        .cumsum()
        .astype(str)
    )

    # join together messages in the same coversation
    conversations = (
        dataframe
        .groupby('conv_num')
        [content_col]
        .apply(lambda x: ' '.join(map(str, x)))
        .str.strip()
        .str.replace('\\s{2,}', ' ', regex=True)
        .reset_index()
    )

    documents = conversations[content_col].tolist()

    return documents

def convert_to_txt(source: str, output: str, timestamp: bool=False):
    pd_data = ms_import_data(source)

    pd_data['clean_content'] = (
        pd_data['content']
        .str.strip()
        .str.replace('[^a-zA-Z0-9\\s]', '', regex=True)
        .str.replace('\\s{2,}', ' ', regex=True)
    )

    if timestamp:
        pd_data['conv_text'] = (
            pd_data['sender_name'] + 
            ' at ' + 
            pd_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S') + 
            ': ' + 
            pd_data['clean_content']
        )
        chat_content = pd_data[['conv_text']]
        chat_content.to_csv(output, index=False, header=False)
    else:
        pd_data['conv_text'] = (
            pd_data['sender_name'] + 
            ': ' + 
            pd_data['clean_content']
        )
        chat_content = pd_data[['conv_text']]
        chat_content.to_csv(output, index=False, header=False)