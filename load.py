import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import math
from keras.preprocessing.text import Tokenizer


def load_files(train_path, test_path):
    train = pd.read_csv(train_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')

    train = train.set_index('train_id')

    return train, test


def handle_missing(df):
    df.category_name.fillna(value="missing", inplace=True)
    df.brand_name.fillna(value="missing", inplace=True)
    df.item_description.fillna(value="missing", inplace=True)
    return df


def encode_categorical_labels(train_df, test_df):
    le = LabelEncoder()

    le.fit(np.hstack([train_df.category_name, test_df.category_name]))
    train_df.category_name = le.transform(train_df.category_name)
    test_df.category_name = le.transform(test_df.category_name)

    le.fit(np.hstack([train_df.brand_name, test_df.brand_name]))
    train_df.brand_name = le.transform(train_df.brand_name)
    test_df.brand_name = le.transform(test_df.brand_name)

    del le

    return train_df, test_df


def text_to_sequences(train_df, test_df):
    raw_text = np.hstack([train_df.item_description.str.lower(), train_df.name.str.lower()])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(raw_text)

    # TODO: clean the text further, including stemming and lemmatization
    train_df["seq_item_description"] = tokenizer.texts_to_sequences(train_df.item_description.str.lower())
    test_df["seq_item_description"] = tokenizer.texts_to_sequences(test_df.item_description.str.lower())

    train_df["seq_name"] = tokenizer.texts_to_sequences(train_df.name.str.lower())
    test_df["seq_name"] = tokenizer.texts_to_sequences(test_df.name.str.lower())

    return train_df, test_df, tokenizer


def save_data():
    return -1