# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:01:32 2016
@author: joomi
"""

# Load in libraries
import gensim
from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim import corpora, models
import pyLDAvis.gensim as gensimvis
import pyLDAvis

# Useful functions for LDA topic modeling using gensim =======================

def make_lda_model(token_list, num_topics, passes=40):
    """
    Function to generate LDA models
    Args:
        token_list: a list of lists, one list for each of our original docs
        num_topics: number of topics
        passes: passes for the LDA model
    Returns:
        dictionary
        corpus: document-term matrix
        lda_model
    """
    dictionary = corpora.Dictionary(token_list)

    # doc2bow() converts dict into bag of words.
    # the result, corpus, is a list of vectors equal to the number of docs.
    corpus = [dictionary.doc2bow(text) for text in token_list]

    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics,
            id2word = dictionary, passes = passes)

    return dictionary, corpus, lda_model


def save_model_elements(dictionary, corpus, lda_model,
                        dict_filename, corpus_filename, lda_filename):
    """
    Function to save model elements generated from make_lda_model
    """
    # store the dict for future reference.
    dictionary.save(dict_filename)
    print("dictionary saved")

    # store corpus
    corpora.McCorpus.serialize(corpus_filename, corpus)
    print("corpus saved")

    # Save the model
    lda.save(lda_filename)
    print("LDA model saved")


def visualize_topics(lda_model, corpus, dictionary):
    """
    Function to visualize topics using pyLDAvis
    """
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.display(vis_data)
