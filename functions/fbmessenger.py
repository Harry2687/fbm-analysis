import os
import json
import pandas as pd
import datetime as dt

def ms_convert_timestamp(timestamp_ms):
    timestamp = dt.datetime.fromtimestamp(timestamp_ms/1000)
    return timestamp

def ms_import_data(directory):
    data_file_names = os.listdir(directory)
    data_files = [os.path.join(directory, data_file_name) for data_file_name in data_file_names]

    messenger_data = pd.DataFrame()

    for data_file in data_files:
        with open(data_file) as file:
            data = json.load(file)
        json_data = pd.json_normalize(data['messages'])
        messenger_data = pd.concat([messenger_data, json_data])

    messenger_data = messenger_data[[
        'sender_name',
        'timestamp_ms',
        'content'
    ]].dropna().sort_values('timestamp_ms', ascending=False)

    messenger_data['timestamp'] = messenger_data['timestamp_ms'].apply(ms_convert_timestamp)

    messenger_data = messenger_data[[
        'sender_name',
        'timestamp',
        'content'
    ]]

    return messenger_data
