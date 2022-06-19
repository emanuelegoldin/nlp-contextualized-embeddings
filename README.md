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

After each execution a PNG file with a graph showing the embeddings of the selected word in each sentence is added into the output folder.