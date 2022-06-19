from utilities import plotEmbeddings, tensor2Numpy
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

BERT_MODEL_PATH = './models/bert'
BERT_INPUT_FILE = './input/bert.csv'
PARAGRAPH = 'Paragraph'
WORD = 'Word'
TOKENIZED = 'Tokenized'
MARKED = 'Marked'

def getTokenizer():
    return BertTokenizer.from_pretrained(BERT_MODEL_PATH)

def toPyTorch(l):
    return torch.tensor([l])

def getBERTModel():
    model = BertModel.from_pretrained(BERT_MODEL_PATH,
                                  output_hidden_states = True,
                                  )
    model.eval()
    return model

def refactor(tkn_emb):
    tkn_emb = torch.squeeze(tkn_emb, dim=1)
    tkn_emb = tkn_emb.permute(1,0,2)
    return tkn_emb

def concatenate(tkn_emb):
    token_vecs_cat = []
    for token in tkn_emb:
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        token_vecs_cat.append(cat_vec)
    return token_vecs_cat

def selectLabelToShow(label, tokens):
    to_show = []
    for token in tokens:
        to_show.append(token == label)
    return to_show

def markText(text):
    return "[CLS]"+text+" [SEP]"

def readInput():
    data = pd.read_csv(BERT_INPUT_FILE)
    data[MARKED] = data[PARAGRAPH].apply(markText)
    return data


def main():
    input = readInput()

    relevant_word = input[WORD].values[0]
    text = input[PARAGRAPH].values[0]
    marked_text = markText(text)

    tokenizer = getTokenizer()
    model = getBERTModel()

    tokenized_text = tokenizer.tokenize(marked_text)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    
    tokens_tensor = toPyTorch(indexed_tokens)
    segments_tensors = toPyTorch(segments_ids)

    with torch.no_grad():
        #outputs.hidden_states[layer][sentence][word]
        outputs = model(tokens_tensor, segments_tensors)

    # Concatenate the tensors for all layers.
    token_embeddings = torch.stack(outputs.hidden_states, dim=0)

    token_embeddings = refactor(token_embeddings)

    token_vecs = concatenate(token_embeddings)
    words_to_show = selectLabelToShow(relevant_word, tokenized_text)

#TODO: extract context as label
    labels = [
        "bank vault",
        "bank robber",
        "river bank"
    ]
    numpy_vecs_cat = tensor2Numpy(token_vecs)
    plotEmbeddings(numpy_vecs_cat, labels, words_to_show, 'bert')

    return

main()