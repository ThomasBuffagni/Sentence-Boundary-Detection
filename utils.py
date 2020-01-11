import numpy as np

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

def display_pred_sentences(words, probs, threshold=0.5):
    """
        Display the predicted sentences according to the probabilities calculated by the model
        and the threshold
    """
    
    for curr_word, prob_dict in zip(words, probs):
        # end = ' ' if prob_dict[0] > prob_dict[1] else '\n\n'
        end = '\n\n' if prob_dict[1] >= threshold else ' '
        print(curr_word, end=end)
    
def wrap_sentences(words, probs, threshold=0.5, start_tag='<span>', end_tag='</span>'):
    html = start_tag

    for curr_word, prob_dict in zip(words, probs):
        end = end_tag + start_tag if prob_dict[1] >= threshold else ' '
        html += curr_word + end
    
    # Remove the last start tag present at the end of the string
    return html[:-len(start_tag)]
