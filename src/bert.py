from utilities import plotEmbeddings, tensor2Numpy
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

BERT_MODEL_PATH = './models/bert'
BERT_INPUT_FILE = './input/bert.csv'
SENTENCE = 'Sentence'
WORD = 'Word'
TOKENIZED = 'Tokenized'
ID = 'Id'
MARKED = 'Marked'
SEGMENT = 'Segment'
TENSOR = 'Token_Tensor'
SGM_TENSOR = 'Segment_Tensor'

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
    data[MARKED] = data[SENTENCE].apply(markText)
    return data

def getLabels(words, target):
    labels = []
    length = len(words)
    for index in range(length):
        if words[index] == target:
            start = max(0, index - 1)
            end = min(length - 1, index + 1)
            label = words[start] + ' ' + target.upper() + ' ' + words[end]
            labels.append(label)
    return labels

def iteration(model, tokens_tensor, segments_tensors, tokenized_text, text, target, plot_id):
    with torch.no_grad():
        #outputs.hidden_states[layer][sentence][word]
        outputs = model(tokens_tensor, segments_tensors)

    # Concatenate the tensors for all layers.
    token_embeddings = torch.stack(outputs.hidden_states, dim=0)

    token_embeddings = refactor(token_embeddings)

    token_vecs = concatenate(token_embeddings)

    words_to_show = selectLabelToShow(target, tokenized_text)

    labels = getLabels(tokenized_text, target)

    numpy_vecs_cat = tensor2Numpy(token_vecs)
    plotEmbeddings(numpy_vecs_cat, labels, words_to_show, 'bert_'+target+'_'+str(plot_id))

    return

def getSegment(tokens):
    return [1] * len(tokens)

def main():
    input = readInput()
    input[MARKED] = input[SENTENCE].apply(markText)

    tokenizer = getTokenizer()
    model = getBERTModel()

    input[TOKENIZED] = input[MARKED].apply(tokenizer.tokenize)
    input[ID] = input[TOKENIZED].apply(tokenizer.convert_tokens_to_ids)
    input[SEGMENT] = input[TOKENIZED].apply(getSegment)

    input[TENSOR] = input[ID].apply(toPyTorch)
    input[SGM_TENSOR] = input[SEGMENT].apply(toPyTorch)
    
    id = 0
    for _, row in input.iterrows():
        iteration(model, row[TENSOR], row[SGM_TENSOR], row[TOKENIZED], row[SENTENCE], row[WORD], id)
        id = id + 1
    return

main()