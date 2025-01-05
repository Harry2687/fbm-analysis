import re
import pandas as pd
import gensim as gs

custom_stopwords = [
    'aditya',
    'gupta',
    'dhruv',
    'jobanputra',
    'harry',
    'zhong',
    'mansoor',
    'khawaja',
    'saquib',
    'ahmed',
    'anand',
    'karna',
    'chaitany',
    'goyal',
    'himal',
    'pandey',
    'anirudth',
    'sanivarapu',
    'sai',
    'roshan',
    'prashant',
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
    'stuff',
    'k',
    'damn',
    'ez',
    'ill',
    'smh',
    'f',
    'ofc',
    'u',
    'tf',
    'nice',
    'wtf',
    'tbf',
    'ngl',
    'm',
    'good',
    'time',
    'think',
    'probs',
    'know',
    'gay',
    'tho',
    'actually',
    'pretty',
    'people',
    'doesnt',
    'need',
    'better',
    'use',
    'isnt',
    'said',
    'oh',
    'shit',
    'thing',
    'cause',
    'look',
    'want',
    'bad', 
    'mean', 
    'going', 
    'maybe', 
    'day', 
    'bit', 
    'theres',
    'guys', 
    'cool', 
    'coz', 
    'way', 
    'tbh', 
    'today', 
    'wait',
    'right', 
    'noice', 
    'literally', 
    'anirudh', 
    'retarded',
    'legit',
    'hes',
    'lot',
    'agreed',
    'whats',
    'year',
    'played',
    'wanna',
    'getting',
    'looks',
    'af',
    'cuz',
    'probably',
    'kinda',
    'went',
    'wont',
    'ask',
    'thought',
    'pre',
    'rn',
    'fine',
    'theyre',
    'havent',
    'basically',
    'solid',
    'ive',
    'x',
    'wasnt',
    'okay',
    'y',
    'week',
    'surely',
    'things',
    'saying',
    'sounds',
    'set',
    'start',
    'makes',
    'lets',
    'guess',
    'ones',
    'youre',
    'apparently',
    'rip',
    'yes',
    'bruh',
    'anyways',
    'gotta',
    'send',
    'tmr',
    'wouldnt',
    'times',
    'btw'
]

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
        'reacted to your message',
        'https'
    ]

    # remove chat actions which aren't part of the conversation
    dataframe = dataframe[
        ~dataframe[cleaned_col]
        .str.contains(
            '|'.join(chat_actions)
        )
    ]

    if rm_stopwords:
        # remove gensim and custom stopwords
        dataframe.loc[:, cleaned_col] = (
            dataframe[cleaned_col]
            .apply(gs.parsing.preprocessing.remove_stopwords)
            .apply(remove_custom_stopwords, args=(custom_stopwords,))
            .str.strip()
            .str.replace('\\s{2,}', ' ', regex=True)
        )

    return dataframe

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
    conversations = conversations[conversations[content_col] != '']

    documents = conversations[content_col].tolist()

    return documents

def top_words(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['word'] = (
        dataframe['clean_content']
        .str.split(' ')
    )

    df_wcount = dataframe[['sender_name', 'word']].explode('word')
    df_wcount = (
        df_wcount[df_wcount['word'] != '']
        .groupby('word')
        .count()
        .rename(columns={'sender_name': 'count'})
        .sort_values('count', ascending=False)
    )

    return df_wcount