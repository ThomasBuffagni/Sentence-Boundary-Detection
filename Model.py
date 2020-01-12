import re
from collections import defaultdict, Counter

from utils import is_word_capitalized

class Model():
    def __init__(self):
        self.training_sentences = None
        self.labeled_grams = None
        self.features = None
        self.test_words = None
        self.pred_probs = None
        

    def fit(self, sentences):
        """
            Train the model

            sentences: a list of sentences. Each sentence is a list of words.
        """
        print("Starting Training...")
        self.training_sentences = sentences
        self.labeled_grams = self.get_3grams(sentences, labeled=True)
        self.features = self.get_features(self.labeled_grams, labeled=True)


    def predict_prob(self, words):
        """
            Calculate the probability of each word to be the end of a sentence.

            words: a list of words
        """
        print("Starting Testing...")
        self.test_words = words
        grams = self.three_grams_from_sentence(words, '~', '~')
        raw_features = self._extract_raw_features(grams, labeled=False)
        self.pred_probs = self.get_probs(grams, raw_features)
        return self.pred_probs

# ============================================= #

    def get_3grams(self, sentences, labeled):
        """ 
        Create a list of 3grams from a list of sentences
        """
        print("Extracting 3grams from sentences...")

        grams = list()
        for i, sentence in enumerate(sentences):
            if i == 0: prev_word = '~'
            else: prev_word = sentences[i-1][-1]

            if i == len(sentences) - 1: next_word = '~'
            else: next_word = sentences[i+1][0]
            
            temp = self.three_grams_from_sentence(sentence, prev_word, next_word)
            
            if labeled is True: temp = self._label_sentence_grams(temp)
            
            grams.extend(temp)
            
        return grams
    

    def three_grams_from_sentence(self, sentence, prev_word, next_word):
        """
        Create a list of 3grams from a sentence (list of words),
        the preceding word, and the following word
        """
        grams = list()

        if len(sentence) < 2:
            return [(prev_word, sentence[0], next_word)]

        # First 3 gram of the sentence
        grams.append((prev_word, sentence[0], sentence[1]))
        
        grams.extend([(sentence[i-1], sentence[i], sentence[i+1]) for i in range(1, len(sentence) - 1)])
        
        # Last 3 gram of the sentence
        grams.append((sentence[-2], sentence[-1], next_word))

        return grams


    def _label_sentence_grams(self, grams):
        """
        Label each 3gram of the list with 0,
        except for the last one (the end of a sentence) with one
        """
        # Label all the others as not the end of the sentence
        labeled = [(g, 0) for g in grams[:-1]]
        # Label the last 3gram as the end of the sentence
        labeled.append((grams[-1], 1))
        return labeled


    def get_features(self, grams, labeled):
        print("Extracting Features...")
        raw_features = self._extract_raw_features(grams, labeled)

        features = dict()
        # Initialize temporary dictionaries for calculation
        total = defaultdict(float)
        feature_dict = defaultdict(Counter)

        # learn the features of each 3gram with its label
        for gram, feats in zip(self.labeled_grams, raw_features):
            context, label = gram

            total[label] += len(context) + len(feats)
            
            # Context <-> Label 
            for i, word in enumerate(context):
                feature_dict[label]['w' + str(i) + '_' + word] += 1
            
            # Arbitrary Features <-> Label 
            for name, val in feats.items():
                feature_dict[label][name + '_' + val] += 1
        
        # Smoothing
        print("Smoothing...")
        smooth_coeff = 0.1
        
        all_features = set()
        for label in total:
            all_features.update(feature_dict[label].keys())
        
        for label, counts in total.items():
            total[label] += len(all_features) * smooth_coeff

            for name in all_features:
                feature_dict[label][name] += smooth_coeff
                feature_dict[label][name] /= total[label]
                features[(label, name)] = feature_dict[label][name]

        # Prior probabiliy for each label
        s = sum(total.values())
        for label, count in total.items():
            features[(label, 'prior')] = count / s
        
        self.labels = list(total.keys())
        return features


    def _extract_raw_features(self, grams, labeled):
        """
        For each 3gram, create a dictionary corresponding to some features
        used by the model.
        Return a list of dictionaries
        """

        # Initialize the list
        features = [0 for _ in range(len(grams))]

        # Create a dictionary for each 3gram
        for i, gram in enumerate(grams):
            # The 3gram may be labeled, extract it if necessary
            if labeled is True: context, label = gram
            else: context = gram
            
            feat = dict()

            # Previous word is capitalized
            feat['prev_word_cap'] = str(is_word_capitalized(context[0]))
            # Length of previous word
            feat['prev_word_len'] = str(len(context[0]))
            # Current word in capitalized
            feat['curr_word_cap'] = str(is_word_capitalized(context[1]))
            # Length of current word
            feat['curr_word_len'] = str(len(context[1]))
            # Next word is capitalized
            feat['next_word_cap'] = str(is_word_capitalized(context[2]))
            # Length of next word
            feat['next_word_len'] = str(len(context[2]))
            
            features[i] = feat
        
        return features


    def group_sentence(self, words, start, end):
        return words[start:end+1]


    def get_probs(self, grams, raw_features):
        probs = list()
        for context, feats in zip(grams, raw_features):
            prob = self.get_prob(context, feats)
            probs.append(prob)
        
        return probs


    def get_prob(self, context, features):
        """
        Return a list of dictionaries.
        Each dictionary corresponds to one gram and contains
        2 probabilities corresponding to 'not an end of a sentence' and 'end of a sentence'
        """
        # Initialize probabilities with the priori probability calculated during training
        label_prob = {label: self.features[(label, 'prior')] for label in self.labels}
        
        for label in self.labels:
            # Evaluate arbitrary features
            for name, val in features.items():
                feature_name = name + '_' + val
                key = (label, feature_name)
                if key in self.features:
                    label_prob[label] *= self.features[key]
            
            # Evaluate context
            for i, word in enumerate(context):
                feature_name = 'w' + str(i) + '_' + word
                key = (label, feature_name)
                if key in self.features:
                    label_prob[label] *= self.features[key]

        
        # Normalization
        total_prob = sum(label_prob.values())
        for label in self.labels:
            label_prob[label] /= total_prob

        return label_prob