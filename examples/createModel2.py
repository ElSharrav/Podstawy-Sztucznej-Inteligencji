import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
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
import sys

#train

script_dir = os.path.dirname(__file__)
data_path_train = os.path.join(script_dir, "../data/clean_train.csv")

train_df = pd.read_csv(data_path_train)

modelInput = train_df[['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']]
modelOutput = train_df[['Class']]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

y = modelOutput['Class'].astype(int)  
y_cat = to_categorical(y, num_classes=11)

X_train, X_test, y_train, y_test = train_test_split(modelInput, y_cat, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

print("Shape: " + str(X_train.shape[1]))

# Build the model
model = Sequential([
    BatchNormalization(),
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.15),
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.15),
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.15),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.15),
    Dense(11, activation='softmax')  # 11 genres
])


print(train_df['Class'].value_counts())


model.compile(
              optimizer=Adam(learning_rate=0.01),
              #optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

y_int = y.values.flatten()
class_weights = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
class_weight_dict = dict(enumerate(class_weights))

model.fit(X_train, y_train, epochs=300, shuffle=True,verbose=2,
          validation_data=(X_test, y_test),
          class_weight=class_weight_dict,
          batch_size=64,
          callbacks=[early_stop]
          )

# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(classification_report(y_true, y_pred, digits=3))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class_names = ['Acoustic/Folk', 'Alt_Music', 'Blues', 'Bollywood', 'Country',
               'HipHop', 'Indie Alt', 'Instrumental', 'Metal', 'Pop', 'Rock']

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', yticklabels = class_names, xticklabels = class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


#model.save("model.h5")