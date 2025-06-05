import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Latent Dirichlet Allocation""")
    return


@app.cell
def _():
    import marimo as mo
    import modules.preprocessing as prep
    import modules.lda.preprocessing as lda_prep
    import modules.lda.models as lda_models
    import gensim as gs
    return lda_models, lda_prep, mo, prep


@app.cell
def _(mo):
    mo.md(r"""Import and clean data.""")
    return


@app.cell
def _(lda_prep, prep):
    chat_data = prep.ms_import_data('data/the_office')
    chat_data = lda_prep.lda_preprocess(chat_data, 'content', 'clean_content')
    return (chat_data,)


@app.cell
def _(chat_data, lda_prep):
    top_words = lda_prep.top_words(chat_data).head(50)
    top_words
    return


@app.cell
def _(mo):
    mo.md(r"""Count number of occurances of specified word by sender.""")
    return


@app.cell
def _(lda_prep, prep):
    chat_data_with_stopwords = prep.ms_import_data('data/the_office')
    chat_data_with_stopwords = lda_prep.lda_preprocess(chat_data_with_stopwords, 'content', 'clean_content', False)
    word_count = prep.sender_wordcount(chat_data_with_stopwords, 'coc')
    word_count
    return


@app.cell
def _(mo):
    mo.md(r"""Split messages into conversations which are separated by at least 10 minutes.""")
    return


@app.cell
def _(chat_data, lda_prep):
    documents = lda_prep.lda_getdocs(chat_data, 'clean_content', 'timestamp')
    return (documents,)


@app.cell
def _(mo):
    mo.md(r"""Run LDA, where documents are the previously defined conversations.""")
    return


@app.cell
def _(documents, lda_models):
    model = lda_models.fbm_lda(documents)
    model.train_lda(7)
    model.print_topics(10)
    return (model,)


@app.cell
def _(model):
    model.get_coherence()
    return


@app.cell
def _(documents, lda_models):
    import pandas as pd

    class topic_plotter(lda_models.fbm_lda):
        def plot(self):
            document_topics = self.lda_model.get_document_topics(self.corpus, minimum_probability=0)
            topic_dist_df = pd.DataFrame(document_topics)
            topic_dist_df = topic_dist_df.map(lambda x: x[1])
            plot = (
                topic_dist_df
                .plot(kind='bar', stacked=True, xticks=[])
                .legend(bbox_to_anchor=(1, 1))
            )
            return plot


    plotter = topic_plotter(documents)
    plotter.train_lda(7)
    plotter.plot()
    return


@app.cell
def _(mo):
    mo.md(r"""Hyperparameter tuning.""")
    return


@app.cell
def _(documents, lda_models):
    tuning = lda_models.tune_lda(documents)
    tuning_results = tuning.tune(2, 20, 1, show_progress=True)
    return


if __name__ == "__main__":
    app.run()
