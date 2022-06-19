from torch import embedding
from utilities import elmo_plot, preprocessing
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import pandas as pd

# Path to csv
ELMO_INPUT_FILE = './input/elmo.csv'
# Columns
PROCESSED_SENTENCE = 'Processed_Sentence'
SENTENCE = 'Sentence'
WORD = 'Word'

def getPad(sentences):
    return len(max(sentences, key=len).split())

def getELMoModel():
    tf.disable_eager_execution()
    #Load pre-trained model
    embed_ = hub.Module("./models/elmo", trainable=True)
    return embed_

def elmo_vectors_sentence(x, model):
  sentence_embeddings = model(x, signature="default", as_dict=True)["elmo"]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    return sess.run(sentence_embeddings)

def readInput():
    data = pd.read_csv(ELMO_INPUT_FILE)
    data[PROCESSED_SENTENCE] = data[SENTENCE].apply(preprocessing)
    return data

def refactoring(sentences, relevant):
    max_length = getPad(sentences)
    word_labels = []
    to_show = []
    for sentence, word_to_focus in zip(sentences, relevant):
        words = sentence.split()
        for word in words:
            to_show.append(word == word_to_focus)
            word_labels.append(word)
        if(len(words) < max_length):
            for i in range(max_length - len(words)):
                word_labels.append('PAD')
    return word_labels, to_show

def getWordEmbedding(elmo_vectors):
    embeddings = []
    for sentence in elmo_vectors:
        for word_embedding in sentence:
            embeddings.append(word_embedding)
    return embeddings

def main():

    input = readInput()
    sentences = input[SENTENCE].values.tolist()

    model = getELMoModel()

    word_labels, to_show = refactoring(input[PROCESSED_SENTENCE], input[WORD])

    elmo_vec = elmo_vectors_sentence(input[PROCESSED_SENTENCE], model)

    embeddings = getWordEmbedding(elmo_vec)

    elmo_plot(embeddings, word_labels, sentences, to_show, 'elmo')
    return

main()