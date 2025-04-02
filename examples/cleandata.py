import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

import pandas as pd
import os
import re


#test


script_dir = os.path.dirname(__file__)
data_path_test = os.path.join(script_dir, "../data/test.csv")
output_path_test = os.path.join(script_dir, "../data/clean_test.csv")


test_df = pd.read_csv(data_path_test)


test_df.dropna(inplace=True)


def is_ascii(s):
    return bool(re.match(r'^[\x00-\x7F]+$', str(s)))


clean_test_df = test_df[test_df["Artist Name"].apply(is_ascii) & test_df["Track Name"].apply(is_ascii)]


clean_test_df.to_csv(output_path_test, index=False)


print(f"Total rows after cleaning: {len(clean_test_df)}")
print(clean_test_df.head())


#train


script_dir = os.path.dirname(__file__)
data_path_train = os.path.join(script_dir, "../data/train.csv")
output_path_train = os.path.join(script_dir, "../data/clean_train.csv")


train_df = pd.read_csv(data_path_train)


train_df.dropna(inplace=True)


def is_ascii(s):
    return bool(re.match(r'^[\x00-\x7F]+$', str(s)))


clean_train_df = train_df[
    train_df["Artist Name"].apply(is_ascii) &
    train_df["Track Name"].apply(is_ascii)
]

clean_train_df.to_csv(output_path_train, index=False)


print(f"Total rows after cleaning: {len(clean_train_df)}")
print(clean_train_df.head())
