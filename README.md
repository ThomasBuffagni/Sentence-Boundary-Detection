# Sentence Boundary Detection


## The Model
Naive Bayes Classification


### Training
The model takes a list of sentences. Each sentence is a list of words.
The model will learn the caracteristics of each word of the sentence, especially at the end. 


### Predictions
The model calculates 2 probabilities for each word:
1. The word is not at the end of a sentence
2. The word is at the end of a sentence


### How it works
#### Training
The model is trained in ... steps:
1. 3grams are extracted from the sentences
2. Each 3gram is labeled: 1 if the central word is at the end of a sentence, 0 otherwise.
3. From each 3gram, some features are extracted, associated to the label of the 3gram and memorized by the model.


#### Predictions
The prediction is composed of ... steps:
1. 3grams are extracted from the list of words
2. From each 3gram, some features are extracted
3. These features are compared to the ones previously learned, and the probabilities are calculated (see https://en.wikipedia.org/wiki/Naive_Bayes_classifier). 