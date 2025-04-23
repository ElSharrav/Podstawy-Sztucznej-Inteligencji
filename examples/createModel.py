import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np

import pandas as pd
import os
import re

#train

script_dir = os.path.dirname(__file__)
data_path_train = os.path.join(script_dir, "../data/clean_train.csv")

train_df = pd.read_csv(data_path_train)

modelInput = train_df[['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature']]
modelOutput = train_df[['Class']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(modelInput)

y = modelOutput['Class'].astype(int)  
y_cat = to_categorical(y, num_classes=11)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(11, activation='softmax')  # 11 genres
])

print(train_df['Class'].value_counts())


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

y_int = y.values.flatten()
class_weights = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
class_weight_dict = dict(enumerate(class_weights))

model.fit(X_train, y_train, epochs=200,
          validation_data=(X_test, y_test),
          class_weight=class_weight_dict,
          #callbacks=[early_stop]
          )

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(classification_report(y_true, y_pred, digits=3))