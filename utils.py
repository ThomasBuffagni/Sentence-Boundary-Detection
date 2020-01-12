import numpy as np
import argparse as ap

def train_test_indices(n):
    a = np.arange(n)
    include_index = np.random.choice(a, size=int(n*0.8), replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[include_index] = True
    return a[mask], a[~mask]

def is_word_capitalized(word):
    """
    Return True is the first letter of word is uppercase
    """
    return word[0].isupper()

    
def pred_sentences(words, probs, threshold=0.5):
    return wrap_sentences(words, probs, threshold, '\n\n', '')

def html_sentences(words, probs, threshold=0.5, start_tag='<span>', end_tag='</span>'):
    return wrap_sentences(words, probs, threshold, start_tag, end_tag)

def wrap_sentences(words, probs, threshold, start_tag, end_tag):
    html = start_tag

    for curr_word, prob_dict in zip(words, probs):
        end = end_tag + start_tag if prob_dict[1] >= threshold else ' '
        html += curr_word + end
    
    # Remove the last start tag present at the end of the string
    return html[:-len(start_tag)].strip()

def to_file(string, file_name):
    print("Writing data on file " + file_name + " ...")
    file = open(file_name, "w")
    file.write(string)

def parse_command_line():
    parser = ap.ArgumentParser()
    parser.add_argument("-fi", "--file", type=str,
                        help="path of the output file")
    parser.add_argument("-fo", "--format", type=str, default="text",
                        help="Format of the output sentences")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="Probability threshold")
    
    args = parser.parse_args()

    return args
