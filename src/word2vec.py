from gensim.models import KeyedVectors
import matplotlib.pyplot as plt 
import numpy as np

WORD2VEC_MODEL_PATHFILE = './models/word2vec/word2vec-google-news-300/word2vec-google-news-300.gz'

def getWord2VecModel():
    return KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATHFILE, binary=True, limit=500000)

def plotCustom(Wv, Wl):
    # setting axis
    b1 = (Wv[1]-Wv[0])
    b2 = (Wv[3]-Wv[2])

    W = np.array(Wv)
    B = np.array([b1,b2])
    Bi = np.linalg.pinv(B.T)

    Wp = np.matmul(Bi,W.T)
    Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T

    plt.figure(figsize=(12,7))
    plt.axvline()
    plt.axhline()
    plt.scatter(Wp[0,:], Wp[1,:])
    rX = max(Wp[0,:])-min(Wp[0,:])
    rY = max(Wp[1,:])-min(Wp[1,:])
    eps = 0.005
    for i, txt in enumerate(Wl):
        plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))

    plt.savefig('output/w2v_custom.png')

def main():
    model = getWord2VecModel()

    # get vectors from model of the following words
    Wl = ['river', 'vault', 'money', 'water', 'bank']
    Wv = []
    for i in range(len(Wl)):
        Wv.append(model[Wl[i]])

    plotCustom(Wv,Wl)
    
    return

main()