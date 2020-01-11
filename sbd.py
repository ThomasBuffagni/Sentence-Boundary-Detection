# http://www.sls.hawaii.edu/bley-vroman/brown_corpus.html

import nltk
import numpy as np
#nltk.download() # this opens a GUI to download all corpora needed
from nltk.corpus import brown

from Model import Model
from utils import display_pred_sentences, train_test_indices, wrap_sentences

if __name__ == "__main__":
    
    sentences = np.array(brown.sents())
    #N = len(sentences)
    N = 50

    train_indices, test_indices = train_test_indices(N)

    # Building Training Data
    train_sentences = sentences[train_indices]    

    # Building Testing Data
    test_sentences = sentences[test_indices]
    test_words = [w for sentence in test_sentences for w in sentence] # unpack the sentences into a list of words

    # Training the model
    model = Model()
    model.fit(train_sentences)

    # Finding sentence boundaries
    pred_probs = model.predict_prob(test_words)
    display_pred_sentences(test_words, pred_probs)
    # print(wrap_sentences(test_words, pred_probs, 0.5, '<span>', '</span>'))
