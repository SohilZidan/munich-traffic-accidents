# This will be replaced with your bucket name after running the `sed` command in the tutorial
BUCKET = "gs://dps-opendataportal-354609-bucket"


import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from google.cloud import storage
import io

# from misc import Normalization

class Normalization:

  fields_to_normalize = ["JAHR", "MONAT"]

  def __init__(self, train_stats):
    train_stats_tmp = train_stats.copy()
    target_stats = train_stats_tmp.loc[:, train_stats_tmp.columns.isin(["WERT"])]
    train_stats_tmp = train_stats_tmp.loc[:, train_stats_tmp.columns.isin(self.fields_to_normalize)]

    self.target_stats = target_stats.transpose()
    self.train_stats = train_stats_tmp.transpose()

  def normalize_features(self, x):
    # return (x - train_stats['mean']) / train_stats['std']
    normed_train_data = x.copy()
    normed_train_data[self.fields_to_normalize] = (normed_train_data[self.fields_to_normalize] - self.train_stats['min']) / (self.train_stats['max']- self.train_stats['min'])
    return normed_train_data

  def normalize_target(self, x):
    # return (x - target_stats['mean']) / target_stats['std']
    return (x - self.target_stats['min']) / (self.target_stats['max']- self.target_stats['min'])

  def denormalize_target(self, x):
    # return x * target_stats['std']["WERT"] + target_stats['mean']["WERT"]
    return x * (self.target_stats['max']["WERT"] - self.target_stats['min']["WERT"]) + self.target_stats['min']["WERT"]


def predict(x, y_true):
    normalized_sample = norm.normalize_features(x).tolist()
    y_pred = norm.denormalize_target(model.predict([normalized_sample]))
    print("GT:", y_true, "\nPredicted:", y_pred)
    print("AE:",abs(y_true-y_pred))

if __name__ == "__main__":
    print(tf.__version__)
    # Read data
    dataset_path = "https://opendata.muenchen.de/dataset/5e73a82b-7cfb-40cc-9b30-45fe5a3fa24e/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7/download/220511_monatszahlenmonatszahlen2204_verkehrsunfaelle.csv"
    dataset = pd.read_csv(dataset_path, na_values = "?")

    #################
    # Preprocessing #
    #################
    dataset.tail()
    dataset.isna().sum()
    dataset = dataset.dropna()
    dataset = dataset[["MONATSZAHL", "AUSPRAEGUNG", "JAHR", "MONAT", "WERT"]]
    # drop 2021 entries
    dataset = dataset.drop(dataset[dataset.JAHR == 2021].index)

    dataset["MONAT"] = dataset["MONAT"].map(lambda x: x[-2:])
    dataset["MONAT"] = dataset["MONAT"].astype(np.int64)

    # """one-hot encode the values of MONATSZAHL and AUSPRAEGUNG """
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

    # split
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    ## normalization
    train_stats = train_dataset.describe()
    train_stats.to_pickle(BUCKET+"/train_stats.pkl")

    norm = Normalization(train_stats)
    train_labels = norm.normalize_target(train_dataset[["WERT"]])
    test_labels = norm.normalize_target(test_dataset[["WERT"]])
    train_dataset.pop('WERT')
    test_dataset.pop('WERT')

    normed_train_data = norm.normalize_features(train_dataset)
    normed_test_data = norm.normalize_features(test_dataset)

    ############
    # Training #
    ############
    def build_model():
        model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=[len(normed_train_data.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(32,),
        layers.Dense(1, activation="sigmoid")
        ])
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
        return model

    model = build_model()

    # model inspection
    model.summary()

    # train
    EPOCHS = 100
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    early_history = model.fit(normed_train_data, train_labels, batch_size=32, 
                        epochs=EPOCHS, validation_split = 0.2,
                        callbacks=[early_stop])


    # predict
    sample = [2021, 1, 1,0,0, 0, 1, 0]
    x = pd.Series(dict(zip(train_dataset.keys(), sample)))
    y_true = 16
    predict(x, y_true)

    sample = [2020, 1, 1,0,0, 0, 1, 0]
    x = pd.Series(dict(zip(train_dataset.keys(), sample)))
    y_true = 28
    predict(x, y_true)

    # Learning curve
    def plot_loss(history):
        fig = plt.figure()
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        # plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)

        fig_to_upload = plt.gcf()
        # Save figure image to a bytes buffer
        buf = io.BytesIO()
        fig_to_upload.savefig(buf, format='png')
        # init GCS client and upload buffer contents
        client = storage.Client()
        # BUCKET+'/train.png'
        bucket = client.get_bucket('dps-opendataportal-354609-bucket')
        blob = bucket.blob('train.png')  
        blob.upload_from_file(buf, content_type='image/png', rewind=True)

    plot_loss(early_history)

    # # Export model and save to GCS
    model.save(BUCKET + '/opendataportal/model')
