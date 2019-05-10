import pandas as pd
import numpy as np
import time

def create_glove_dict(glove_file="glove.6B.300d.txt"):
    d = load_glove(glove_file)
    insert_unknown_token(d)
    return d

def load_glove(glove_file="glove.6B.300d.txt"):
    print("Starting to Load Glove vectors") 
    start_time = time.time()
    f = open(glove_file,'r')
    print("Completed opening file in {} seconds".format((time.time() - start_time)))
    glove = {}
    i = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        try:
            v = np.array([float(x) for x in splitLine[1:]])
        except Exception as e:
            print("failed on word {}".format(word))
            print(e)
        glove[word] = v
        if i % 500000 == 0:
            print("Completed reading {} words in {} seconds".format(i, (time.time() - start_time)))
        i += 1
    f.close()
    print("Completed constructing dict in {} seconds".format((time.time() - start_time)))
    return glove

def get_word(d, word):
    if word not in d:
        return d['unknown token']
    else:
        return d[word]

def make_unknown_token(d):
    print("Making Unknown Token")
    start_time = time.time() 
    sum = np.zeros(300, dtype=np.float32)
    for word in d:
        sum += np.asarray(d[word], dtype=np.float32)
    avg = sum / len(d)
    print("Completed Unknown Token in {} seconds".format(time.time() - start_time))
    return avg

def insert_unknown_token(d):
    d['unknown token'] = make_unknown_token(d)

def glove_lookup_test(d, k):
    start_time = time.time()
    v = d[k]
    print("Found vector {} in {} seconds".format(k, (time.time() - start_time)))
    start_time = time.time()
    if "csasddv  asdadfa" not in d:
        print("Unknown processed in {} seconds".format((time.time() - start_time)))

