import os
import json
import pandas as pd

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