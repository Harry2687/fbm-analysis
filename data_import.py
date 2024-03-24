import pandas as pd
import json
import os

data_directory = 'data/message_data'
data_file_names = os.listdir(data_directory)
data_files = [os.path.join(data_directory, data_file_name) 
             for data_file_name in data_file_names]

messenger_data = pd.DataFrame()

for data_file in data_files:
    with open(data_file) as file:
        data = json.load(file)
    json_data = pd.json_normalize(data['messages'])
    messenger_data = pd.concat([messenger_data, json_data])

messenger_data.to_csv('data.csv')