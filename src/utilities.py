from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import string

import nltk

try:
    nltk.data.find('stopwords')
    nltk.data.find('punkt')
    nltk.data.find('wordnet')
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

RANDOM_STATE = 1
ITERATIONS = 2500
NUMBER_OF_COMPONENTS = 2

def plotEmbeddings(vecs, labels, to_show, filename:str):    
    reducedData = reduce(vecs)
    if len(to_show) > 0:
        reducedData = filter(reducedData, to_show)
    pca_plot(reducedData,labels, filename)

def reduce(X):    
    pca_model = PCA(n_components=NUMBER_OF_COMPONENTS, random_state=RANDOM_STATE)
    return pca_model.fit_transform(X)

def filter(X, filter):
    result = []
    for value, toKeep in zip(X, filter):
        if toKeep:
            result.append(value)
    return result

def pca_plot(X, labels, filename):
    x = []
    y = []
    for value in X:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig('output/'+filename+'.png')
    #plt.show(block=True)

def tensor2Numpy(vectors):
    result = []
    for vector in vectors:
        result.append(vector.numpy())
    return result


def text_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def tokenize(text):
    text = word_tokenize(text)
    return text

# remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text

lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

def preprocessing(text):
    text = text_lowercase(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = ' '.join(text)
    return text

def elmo_plot(vecs, tkn_text, labels, to_show, filename:str):
    tokens = []

    for vector in zip(vecs, tkn_text):
        if(not vector[1] == 'PAD'):                 # we won't consider the added zero to match the longest sentence.
            tokens.append(vector[0])
            
    X = reduce(tokens)

    if(len(to_show) > 0):
        X = filter(X, to_show)
    
    pca_plot(X, labels, filename)