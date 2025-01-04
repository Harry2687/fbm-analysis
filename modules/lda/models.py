import gensim as gs
import numpy as np
import pandas as pd
import time

class fbm_lda:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc.split() for doc in documents]
        self.dictionary = gs.corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
    
    def train_lda(self, num_topics, passes=10, random_state=2687):
        self.num_topics = num_topics

        self.lda_model = gs.models.ldamodel.LdaModel(
            corpus=self.corpus,
            num_topics=num_topics,
            id2word=self.dictionary,
            passes=passes,
            random_state=random_state
        )

        return self.lda_model

    def print_topics(self, num_words=5):
        for topic in self.lda_model.print_topics(num_topics=self.num_topics, num_words=num_words):
            print(topic)

    def get_coherence(self, coherence_type='u_mass'):
        coherence = gs.models.CoherenceModel(
            model=self.lda_model,
            texts=self.texts,
            dictionary=self.dictionary,
            coherence=coherence_type
        )

        return coherence.get_coherence()
    
class tune_lda(fbm_lda):
    def tune(self, n_start, n_stop, step, sort: bool=False):
        n_topics_values = np.arange(n_start, n_stop, step)

        tuning_results = pd.DataFrame(
            columns=[
                'n_topics',
                'umass_coherence'
            ]
        )
        
        start_time = time.time()
        for index in range(len(n_topics_values)):
            n_topics = n_topics_values[index]
            lda_model = self.train_lda(num_topics=n_topics)
            umass_coherence = self.get_coherence()

            tuning_results.loc[index] = (
                [n_topics] + 
                [umass_coherence]
            )
        end_time = time.time()
        self.tuning_time = end_time - start_time

        if sort:
            tuning_results = tuning_results.sort_values('umass_coherence', ascending=False)

        return tuning_results