import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score


f_ = './ytb_all.csv'


def ReadCSV(f_path):
    df = pd.read_csv(f_path, delimiter=',', encoding='latin-1')
    X = df.CONTENT
    Y = df.CLASS
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    return X, Y

def ReadCSV_(f_path):
    df = pd.read_csv(f_, delimiter=',', encoding='latin-1')
    text_messages = df.CONTENT
    classes = df.CLASS
    le = LabelEncoder()
    classes = le.fit_transform(classes)
    return text_messages, classes

def memGen(dim=10000, num_char=37):
    dictMem = np.random.randint(2, size=(num_char, dim), dtype='int32')
    dictMem[dictMem == 0] = -1
    return dictMem


def encode(msg, dictMem, dim=10000):
    HV = np.zeros(dim, dtype='int32')
    letter_idx = 0
    for letter in msg:
        if letter == -1:
            np.roll(HV, 500)
        else:
            HV = np.add(HV, dictMem[letter])
        letter_idx += 1

    HV_avg = np.average(HV)
    HV[HV > HV_avg] = 1
    HV[HV < HV_avg] = -1
    HV[HV == HV_avg] = 0
    return HV


def train(X, Y, dictMem, dim, n_class, alpha):
    refMem = np.zeros((n_class, dim), dtype='int32')
    msg_idx = 0
    for msg in X:
        HV = encode(msg, dictMem, dim)
        refMem[Y[msg_idx]] = np.add(refMem[Y[msg_idx]], alpha * HV)
        msg_idx += 1
    return refMem


def test(X, Y, dictMem, refMem, dim=10000):
    Y_pred = []
    msg_idx = 0
    for msg in X:
        HV = encode(msg, dictMem, dim=dim)
        sim = [0, 0]
        sim[0] = cosine_similarity([HV, refMem[0]])[0][1]
        sim[1] = cosine_similarity([HV, refMem[1]])[0][1]
        if sim[0] > sim[1]:
            Y_pred.append(0)
        else:
            Y_pred.append(1)
        msg_idx += 1

    return accuracy_score(Y, Y_pred)


def retrain(X, Y, dictMem, refMem, dim, alpha):
    msg_idx = 0
    for msg in X:
        HV = encode(msg, dictMem, dim)
        sim = [0, 0]
        sim[0] = cosine_similarity([HV, refMem[0]])[0][1]
        sim[1] = cosine_similarity([HV, refMem[1]])[0][1]
        if sim[0] > sim[1]:
            y_pred = 0
        else:
            y_pred = 1
        if y_pred != Y[msg_idx]:
            refMem[Y[msg_idx]] = np.add(refMem[Y[msg_idx]], alpha * HV)
            refMem[y_pred] = np.subtract(refMem[y_pred], alpha * HV)
        msg_idx += 1
    return refMem


dim = 10000
num_char = 37
num_epoch = 20
num_class = 2

downsample = None

if __name__ == '__main__':
    X, Y = ReadCSV(f_)
    X_ = []
    #print(len(X))

    for item in X:
        token_item = []
        for letter in item.lower():
            # print(letter)
            if ord(letter) >= ord('a') and ord(letter) <= ord('z'):
                token_item.append(ord(letter) - ord('a') + 11)
            elif ord(letter) >= ord('0') and ord(letter) <= ord('9'):
                token_item.append(ord(letter) - ord('0') + 1)
            elif letter == ' ':
                token_item.append(-1)
            else:
                pass
                #token_item.append(0)
        X_.append(token_item)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_, Y, test_size=0.1, random_state=19720)
    
    X_train = X_train[:downsample]
    Y_train = Y_train[:downsample]

    dictMem = memGen(dim=dim, num_char=num_char)
    # print(dictMem.shape)
    # input()
    refMem = train(X_train, Y_train, dictMem, dim, num_class, alpha = num_epoch)
    print(cosine_similarity([refMem[0], refMem[1]]))
    acc = test(X_test, Y_test, dictMem, refMem, dim)
    print(acc)

    for epoch in range(num_epoch):
        refMem = retrain(X_train, Y_train, dictMem, refMem, dim, alpha = num_epoch - epoch)
        acc = test(X_test, Y_test, dictMem, refMem, dim)
        print(acc)
