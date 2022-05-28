import os
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization, Embedding
from keras import layers
import sentiment_model
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow_addons.metrics import FBetaScore, F1Score
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sentiment_model import app_config
from sentiment_model.utils.help_func import create_tensorboard_callback

hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

ABS_DIR_PATH = os.path.dirname(os.path.abspath(sentiment_model.__file__))
SAVE_DIR_PATH = os.path.join(ABS_DIR_PATH, app_config["logs_save_dir_path"])


class TfModel(ABC):
    def __init__(self,
                 train_sent,
                 train_labels,
                 val_sent,
                 val_labels,
                 epochs_num=1,
                 loss_func="focal",
                 metric_score="fb"):

        self.train_sent = train_sent
        self.train_labels = train_labels
        self.val_sent = val_sent
        self.val_labels = val_labels
        self.epochs_num = epochs_num

        if loss_func == "focal":
            self.loss_func = tfa.losses.SigmoidFocalCrossEntropy()
        else:
            self.loss_func = "binary_crossentropy"
        if metric_score == "fb":
            self.metric_score = FBetaScore(num_classes=2, average="micro", beta=3.0, threshold=0.5)
        else:
            self.metric_score = F1Score(num_classes=2, average="micro", threshold=0.5)

    @abstractmethod
    def x_layers(self):
        pass

    @abstractmethod
    def model_name(self):
        pass

    @abstractmethod
    def model_experiment_name(self):
        pass

    @abstractmethod
    def embedding_name(self):
        pass

    def model(self):
        inputs = layers.Input(shape=(1,), dtype="string")
        text_vectorizer = self.text_vectorizer()
        embedding = self.embedding()
        x = text_vectorizer(inputs)
        x = embedding(x)
        x = self.x_layers()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs, name=self.model_name())

        model.compile(loss=self.loss_func,
                      optimizer=tf.keras.optimizers.Adam(),
                      # metrics=["accuracy"])
                      metrics=[self.metric_score])
        # print(model.summary())

        model.fit(self.train_sent,
                  self.train_labels,
                  epochs=self.epochs_num,
                  validation_data=(self.val_sent, self.val_labels),
                  callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR_PATH,
                                                         experiment_name=self.model_experiment_name())])

        return model

    def text_vectorizer(self):
        text_vectorizer = TextVectorization(max_tokens=10000,
                                            standardize="lower_and_strip_punctuation",  # how to process text
                                            split="whitespace",  # how to split tokens
                                            ngrams=None,  # create groups of n-words?
                                            output_mode="int",  # how to map tokens to numbers
                                            output_sequence_length=15)

        text_vectorizer.adapt(self.train_sent)
        return text_vectorizer

    def embedding(self):
        embedding = Embedding(input_dim=10000,  # set input shape
                              output_dim=128,  # set size of embedding vector
                              embeddings_initializer="uniform",  # default, intialize randomly
                              input_length=15,
                              name=self.embedding_name())
        return embedding


class Model1(TfModel):

    def x_layers(self):
        x = layers.GlobalAveragePooling1D()
        return x

    def model_name(self):
        return "model_1_dense"

    def model_experiment_name(self):
        return "simple_dense_model"

    def embedding_name(self):
        return "embedding_1"


class Model2(TfModel):

    def x_layers(self):
        x = layers.LSTM(64)
        return x

    def model_name(self):
        return "model_2_LSTM"

    def model_experiment_name(self):
        return "LSTM"

    def embedding_name(self):
        return "embedding_2"


class Model3(TfModel):

    def x_layers(self):
        x = layers.GRU(64)
        return x

    def model_name(self):
        return "model_3_GRU"

    def model_experiment_name(self):
        return "GRU"

    def embedding_name(self):
        return "embedding_3"


class Model4(TfModel):

    def x_layers(self):
        x = layers.Bidirectional(layers.LSTM(64))
        return x

    def model_name(self):
        return "model_4_Bidirectional"

    def model_experiment_name(self):
        return "bidirectional_RNN"

    def embedding_name(self):
        return "embedding_4"


class Model5(TfModel):

    def x_layers(self):
        x = layers.GlobalMaxPool1D()
        return x

    def model_name(self):
        return "model_5_Conv1D"

    def model_experiment_name(self):
        return "Conv1D"

    def embedding_name(self):
        return "embedding_5"


class Model6(TfModel):

    def x_layers(self):
        sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                                input_shape=[],  # shape of inputs coming to our model
                                                dtype=tf.string,  # data type of inputs coming to the USE layer
                                                trainable=False,
                                                # keep the pretrained weights (we'll create a feature extractor)
                                                name="USE")
        return sentence_encoder_layer

    def model_name(self):
        return "model_6_USE"

    def model_experiment_name(self):
        return "tf_hub_sentence_encoder"

    def embedding_name(self):
        return "embedding_6"

    def model(self):
        model = tf.keras.Sequential(
            [self.x_layers(),
             layers.Dense(64, activation="relu"),
             layers.Dense(1, activation="sigmoid")],
            name=self.model_name())
        model.compile(loss=self.loss_func,
                      optimizer=tf.keras.optimizers.Adam(),
                      # metrics=self.metric_score)
                      metrics=["accuracy"])
        # print(model.summary())
        model.fit(self.train_sent,
                  self.train_labels,
                  epochs=self.epochs_num,
                  validation_data=(self.val_sent, self.val_labels),
                  callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR_PATH,
                                                         experiment_name=self.model_experiment_name())])

        return model


class Model7(Model6):

    def model_name(self):
        return "model_7_USE"

    def model_experiment_name(self):
        return "10_percent_tf_hub_sentence_encoder"

    def embedding_name(self):
        return "embedding_7"

    def model(self):
        train_sent_90_per, train_sent_10_per, train_lab_90_per, train_lab_10_per = train_test_split(
            np.array(self.train_sent),
            self.train_labels,
            test_size=0.1,
            random_state=42)

        model = tf.keras.Sequential([self.x_layers(),
                                     layers.Dense(64, activation="relu"),
                                     layers.Dense(1, activation="sigmoid")
                                     ], name=self.model_name())

        model.compile(loss=self.loss_func,
                      optimizer=tf.keras.optimizers.Adam(),
                      # metrics=self.metric_score)
                      metrics=["accuracy"])
        # print(model.summary())
        model.fit(x=train_sent_10_per,
                  y=train_lab_10_per,
                  epochs=self.epochs_num,
                  validation_data=(self.val_sent, self.val_labels),
                  callbacks=[create_tensorboard_callback(SAVE_DIR_PATH, self.model_experiment_name())])

        return model


class Model0:

    def model(self, train_sent, train_lab):
        model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])
        model.fit(train_sent, train_lab)
        return model
