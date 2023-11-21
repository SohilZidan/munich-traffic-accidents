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
    dataset = dataset.drop(dataset[dataset.JAHR > 2020].index)

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

    # Scales and Offsets of Rescale layer
    # input
    target_stats = train_stats.pop("WERT")
    train_stats = train_stats.transpose()
    scale = 1.0/(train_stats["max"] - train_stats["min"]).to_numpy()
    offset = (train_stats["min"]/(train_stats["max"] - train_stats["min"])).to_numpy()
    # output
    target_stats = target_stats.transpose()
    target_scale = (target_stats["max"] - target_stats["min"])
    target_offset = target_stats["min"]

    ############
    # Training #
    ############
    def build_model():
        model = keras.Sequential([
        layers.InputLayer(input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Rescaling (scale, -offset),
        layers.Dense(32, activation='relu'),
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

    early_history = model.fit(train_dataset, train_labels, batch_size=32, 
                        epochs=EPOCHS, validation_split = 0.2, verbose=0,
                        callbacks=[early_stop])

    plot_loss(early_history)

    end2end_model = keras.Sequential(
        [
        layers.InputLayer(input_shape=[len(train_dataset.keys())]),
        model,
        tf.keras.layers.Rescaling(target_scale, target_offset)
        ])

    samples = [
        [2021, 1, 1,0,0, 0, 1, 0],
        [2020, 1, 1,0,0, 0, 1, 0]
    ]
    y_true = [16, 28]
    print(end2end_model.predict(samples))

    # # Export model and save to GCS
    end2end_model.save(BUCKET + '/opendataportal/end2endmodel')
