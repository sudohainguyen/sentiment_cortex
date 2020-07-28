import os
import re
from time import time
import base64
import boto3
from botocore import UNSIGNED
from botocore.client import Config

import numpy as np
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

import underthesea as UTS
import nltk

from models import *
from metrics import *
from utils import *

nltk.download('punkt')

get_custom_objects().update({'Sinusoidal_Position_Embedding': Sinusoidal_Position_Embedding})
get_custom_objects().update({'gelu': Activation(gelu)})
get_custom_objects().update({'ScaledDotProductAttention': ScaledDotProductAttention})
get_custom_objects().update({'ModMultiHeadAttention': ModMultiHeadAttention})
get_custom_objects().update({'LayerNormalization': LayerNormalization})
get_custom_objects().update({'FeedForward': FeedForward})
get_custom_objects().update({'binary_focal_loss_fixed': binary_focal_loss()})
get_custom_objects().update({'f1': f1})


def preprocessing(sentences, word_idx_map, max_l):
    preprocessed_sentences = []
    for sen in sentences:
        sen = sen.replace('..', '.')
        sen = UTS.word_tokenize(sen, format='text')
        sen_split = nltk.sent_tokenize(sen)
        temp = []
        for tmp in sen_split:
            if tmp:
                temp.append(np.asarray(sent2idx(tmp.split(), word_idx_map, max_l)))
            else:
                temp.append(None)
        preprocessed_sentences.append(temp)
    return preprocessed_sentences


class PythonPredictor:
    def __init__(self, config):
        # bucket, key = re.match("s3://(.+?)/(.+)", config["model"]).groups()
        # if os.environ.get("AWS_ACCESS_KEY_ID"):
        #     s3 = boto3.client("s3")  # client will use your credentials if available
        # else:
        #     s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))  # anonymous client

        # model_path = os.path.join('/tmp', 'model.hdf5')
        # s3.download_file(bucket, key, model_path)

        info = readinfo('./sentiment_data/info.tw')

        
        self.word_idx_map = info['vocab']
        self.max_l = info['max_l']
        self.vocab_size = len(self.word_idx_map)
        # kernel_initializer = init_kernel_initializer('glorot',
        #                                              kernel_initializer_normal=False,
        #                                              kernel_initializer_uniform=True)
        # self.model = builder(self.max_l, len(self.word_idx_map), kernel_initializer)
        # self.model.load_weights('../model.hdf5')
        # self.model = load_model('/tmp/model.hdf5')
        self.model = load_model('../model.hdf5')
        self.model._make_predict_function() 
        self.labels = ['neg', 'pos']

    def predict(self, payload):
        sentences = payload['sentences']
        results = []
        preprocessed_sentences = preprocessing(sentences, self.word_idx_map, self.max_l)
        for sen in preprocessed_sentences:
            prediction = self.make_prediction(sen)
            results.append(prediction)
        return results

    def make_prediction(self, sentence):
        # for sen in sentences:
        prediction = {'neg': 0, 'pos': 0}
        for subsen in sentence:
            if subsen is None:
                continue
            subsen[subsen == -1] = self.vocab_size
            subsen = np.expand_dims(subsen, 0)
            logits = self.model.predict(subsen)
            prediction[self.labels[np.argmax(logits, axis=1)[0]]] += 1
        return prediction


if __name__ == "__main__":
    import json
    from time import time
    a = PythonPredictor(None)
    with open('sample.json') as f:
        _json = json.load(f)
    start = time()
    a.predict(_json)
    print(time() - start)
