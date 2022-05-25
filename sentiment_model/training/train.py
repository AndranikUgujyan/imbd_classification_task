import os
import numpy as np
import pandas as pd
import sentiment_model
from matplotlib import pyplot as plt
import dataframe_image as dfi
from sentiment_model import app_config
from sentiment_model.training.evaluation import ModelPrediction
from sentiment_model.training.models import Model0, Model1, Model2, Model3, Model4, Model5, Model6, Model7
import tensorflow as tf

from sentiment_model.utils.help_func import calculate_results

abs_dir_path = os.path.dirname(os.path.abspath(sentiment_model.__file__))

ALL_MODELS_RESULT_PLOT_ABS_PATH = os.path.join(abs_dir_path, app_config["all_models_result_plot_path"])

train_data_abs_path = os.path.join(abs_dir_path, app_config['train_data_path'])
val_data_abs_path = os.path.join(abs_dir_path, app_config['val_data_path'])
test_data_abs_path = os.path.join(abs_dir_path, app_config['test_data_path'])

train_df = pd.read_csv(train_data_abs_path)
val_df = pd.read_csv(val_data_abs_path)
test_df = pd.read_csv(test_data_abs_path)

train_sentences = train_df["review"].to_numpy()
train_labels = train_df["sentiment"].to_numpy()

val_sentences = val_df["review"].to_numpy()
val_labels = val_df["sentiment"].to_numpy()

test_sentences = test_df["review"].to_numpy()
test_labels = test_df["sentiment"].to_numpy()

model_0 = Model0().model(train_sentences, train_labels)
tf_model_1 = Model1(train_sentences, train_labels, val_sentences, val_labels).model()
tf_model_2 = Model2(train_sentences, train_labels, val_sentences, val_labels).model()
tf_model_3 = Model3(train_sentences, train_labels, val_sentences, val_labels).model()
tf_model_4 = Model4(train_sentences, train_labels, val_sentences, val_labels).model()
tf_model_5 = Model5(train_sentences, train_labels, val_sentences, val_labels).model()
tf_model_6 = Model6(train_sentences, train_labels, val_sentences, val_labels).model()
tf_model_7 = Model7(train_sentences, train_labels, val_sentences, val_labels).model()

mp = ModelPrediction(test_sentences, test_labels)

baseline_results = mp.pred(model_0, "naive_bayes")
model_1_results = mp.pred(tf_model_1, "simple_dense")
model_2_results = mp.pred(tf_model_2, "lstm")
model_3_results = mp.pred(tf_model_3, "gru")
model_4_results = mp.pred(tf_model_4, "bidirectional")
model_5_results = mp.pred(tf_model_5, "conv1d")
model_6_results = mp.pred(tf_model_6, "tf_hub_sentence_encoder")
model_7_results = mp.pred(tf_model_7, "tf_hub_10_percent_data")

all_model_results = pd.DataFrame({"baseline": baseline_results,
                                  "simple_dense": model_1_results,
                                  "lstm": model_2_results,
                                  "gru": model_3_results,
                                  "bidirectional": model_4_results,
                                  "conv1d": model_5_results,
                                  "tf_hub_sentence_encoder": model_6_results,
                                  "tf_hub_10_percent_data": model_7_results})
all_model_results = all_model_results.transpose()
print(all_model_results)
all_model_results["accuracy"] = all_model_results["accuracy"] / 100

all_models_result_plot_path = ALL_MODELS_RESULT_PLOT_ABS_PATH.format("all_models_results_under_sampled.png")
all_models_result = all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))
plt.savefig(all_models_result_plot_path, dpi=300)

all_models_results_f1_score_path = ALL_MODELS_RESULT_PLOT_ABS_PATH.format("all_models_f1_score_under_sampled.png")
all_model_results_fig = all_model_results.sort_values("f1", ascending=False)["f1"].plot(kind="bar", figsize=(10, 7))
plt.savefig(all_models_results_f1_score_path, dpi=300)

# get the prediction probabilities from baseline model
baseline_pred_probs = np.max(model_0.predict_proba(val_sentences), axis=1)
model_2_pred_probs = tf.squeeze(tf_model_2.predict(val_sentences), axis=1)
model_6_pred_probs = tf.squeeze(tf_model_6.predict(val_sentences))
combined_pred_probs = baseline_pred_probs + model_2_pred_probs + model_6_pred_probs
combined_preds = tf.round(combined_pred_probs / 3)
print(combined_preds[:20])
ensemble_results = calculate_results(val_labels, combined_preds)
print(ensemble_results)
all_model_results.loc["ensemble_results"] = ensemble_results
all_model_results.loc["ensemble_results"]["accuracy"] = all_model_results.loc["ensemble_results"]["accuracy"] / 100
print(all_model_results)
df_styled = all_model_results.style.background_gradient()

all_models_results_path = ALL_MODELS_RESULT_PLOT_ABS_PATH.format("all_models_f1_score_df_under_sampled.png")
dfi.export(df_styled, all_models_results_path)

model_1_abs_path = os.path.join(abs_dir_path, app_config['model_1_path'])
model_2_abs_path = os.path.join(abs_dir_path, app_config['model_2_path'])
model_3_abs_path = os.path.join(abs_dir_path, app_config['model_3_path'])
model_4_abs_path = os.path.join(abs_dir_path, app_config['model_4_path'])
model_5_abs_path = os.path.join(abs_dir_path, app_config['model_5_path'])
model_6_abs_path = os.path.join(abs_dir_path, app_config['model_6_path'])
model_7_abs_path = os.path.join(abs_dir_path, app_config['model_7_path'])

# tf_model_1.save(model_1_abs_path)
# tf_model_2.save(model_2_abs_path)
# tf_model_3.save(model_3_abs_path)
# tf_model_4.save(model_4_abs_path)
# tf_model_5.save(model_5_abs_path)
# tf_model_6.save(model_6_abs_path)
tf_model_7.save(model_7_abs_path)

