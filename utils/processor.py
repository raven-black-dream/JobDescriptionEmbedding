from bs4 import BeautifulSoup
import numpy as np
import os
import pandas as pd
import spacy
import tensorflow as tf
from tqdm.auto import tqdm


class DataProcessor:

    def __init__(self, nlp, dataset=None) -> None:
        self.data = []
        self.dataset = dataset
        self.labels = []
        self.label_encoding = {}
        self.nlp = nlp

    def load_data(self, fp:str) -> None:

        for root, subdirs, files in os.walk(fp):
            i = 0
            for fn in tqdm(files):
                path = os.path.join(root, fn)
                df = pd.read_csv(path)
                docs = df['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
                vecs = [self.nlp(doc).vector for doc in docs]
                label = fn.split('.')[0]
                labels = [i for j in range(len(vecs))]
                self.data.extend(vecs)
                self.labels.extend(labels)
                self.label_encoding[label] = i
                i += 1

    def process_data(self) -> tf.data.Dataset:
        self.dataset = tf.data.Dataset.from_tensor_slices((self.data, self.labels))
        self.dataset.save('data/dataset')
        self.data = None


    def check_tensors(self):
        lengths = {}
        for row in tqdm(self.data):
            lngth = len(row[0])
            if not lngth in lengths.keys():
                lengths[lngth] = 1
                continue
            else:
                lengths[lngth] += 1
        print(lengths)
        print(max(lengths.keys()))

    def test_train_split(self, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True):
        assert (train_split + test_split + val_split) == 1

        ds_size = len(self.labels)
        self.labels = None

        if shuffle:
            # Specify seed to always have the same split distribution between runs
            ds = self.dataset.shuffle(self.dataset.cardinality(), seed=12)
        else:
            ds = self.dataset
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds
    
if __name__ == '__main__':

    nlp = spacy.load('en_core_web_lg')
    processor = DataProcessor(nlp)
    processor.load_data('data')
    processor.process_data()
    train, val, test = processor.test_train_split()