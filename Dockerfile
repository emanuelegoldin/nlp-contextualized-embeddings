FROM python:3.10-slim-buster
WORKDIR /app

RUN apt-get update && apt-get install -y make g++ wget git git-lfs

COPY ./src/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir ./models
RUN mkdir ./output
RUN mkdir ./models/elmo

#Download ELMo model
RUN wget -c https://tfhub.dev/google/elmo/3?tf-hub-format=compressed -O elmo.tar.gz && \
    tar xzf elmo.tar.gz -C ./models/elmo && \
    rm elmo.tar.gz

#Download BERT model
RUN git clone https://huggingface.co/bert-base-uncased ./models/bert
RUN cd ./models/bert && \
    git lfs install && git lfs pull
RUN rm -r ./models/bert/.git ./models/bert/.gitattributes ./models/bert/README.md

#Download Word2Vec model
ENV GENSIM_DATA_DIR=/app/models/word2vec
RUN cd ./models && \
    python -m gensim.downloader --download word2vec-google-news-300

COPY ./src ./
COPY ./input ./input

CMD [ "/bin/sh" ]