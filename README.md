## Contextual Embeddings

This project show the differences between ELMo, BERT and Word2Vec (used as a baseline) w.r.t. the word embeddings they generate.<br>

The sentences used as input are stored in CSV files with the following structure:
* Sentence: the whole sentence to send to the model
* Word: the word of which embedding we are interested<br>

Move into root directory and build it with:
```bash
dokcer-compose build
```
It will require some time since it will download all the three models.
<br>

---

## How to use it
```bash
docker-compose run mycontainer
```
The previous command will give you access to the bash within the container.<br>
You can then execute each individual model, i.e.:
 
```bash
python bert.py 
python elmo.py
python word2vec.py
```

After each execution a PNG file with a graph showing the embeddings of the selected word in each sentence is added into the output folder.<br>

---
## Models
* [BERT](https://huggingface.co/transformers/v3.0.2/model_doc/bert.html)
* [ELMo](https://tfhub.dev/google/elmo/3)
* [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html?module-gensim.models.word2vec) 

---

## Literature
We used the following article/project as base of our code.
1. [Visualisation of embedding relations (word2vec, BERT)](https://towardsdatascience.com/visualisation-of-embedding-relations-word2vec-bert-64d695b7f36)
2. [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)
3. [Visualizing ELMo Contextual Vectors](https://towardsdatascience.com/visualizing-elmo-contextual-vectors-94168768fdaa)