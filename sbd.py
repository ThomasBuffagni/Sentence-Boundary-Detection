# http://www.sls.hawaii.edu/bley-vroman/brown_corpus.html

import nltk
import numpy as np
#nltk.download() # this opens a GUI to download all corpora needed
from nltk.corpus import brown

from Model import Model
from utils import train_test_indices, html_sentences, pred_sentences, to_file, parse_command_line

if __name__ == "__main__":
    
    args = parse_command_line()

    print("Loading Dataset...")
    sentences = np.array(brown.sents())
    N = len(sentences)

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

    if args.format == "text":
        format = pred_sentences
    elif args.format == "html":
        format = html_sentences
    else:
        print("Unknown format, set to default value")
        format = pred_sentences
        
    
    if args.file:
        to_file(format(test_words, pred_probs, args.threshold), args.file)
    else:
        print(format(test_words, pred_probs, args.threshold))
