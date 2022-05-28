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
from sentiment_model.utils.help_func import calculate_results, compare_baseline_to_new_results

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--models', type=str)
parser.add_argument('--data', type=str)

opts = parser.parse_args()

ABS_DIR_PATH = os.path.dirname(os.path.abspath(sentiment_model.__file__))

ALL_MODELS_RESULT_PLOT_ABS_PATH = os.path.join(ABS_DIR_PATH, app_config["all_models_result_plot_path"])

model_1_abs_path = os.path.join(ABS_DIR_PATH, app_config['model_1_path'])
model_2_abs_path = os.path.join(ABS_DIR_PATH, app_config['model_2_path'])
model_3_abs_path = os.path.join(ABS_DIR_PATH, app_config['model_3_path'])
model_4_abs_path = os.path.join(ABS_DIR_PATH, app_config['model_4_path'])
model_5_abs_path = os.path.join(ABS_DIR_PATH, app_config['model_5_path'])
model_6_abs_path = os.path.join(ABS_DIR_PATH, app_config['model_6_path'])
model_7_abs_path = os.path.join(ABS_DIR_PATH, app_config['model_7_path'])


# train_data_abs_path = os.path.join(ABS_DIR_PATH, app_config['train_data_path'])
# val_data_abs_path = os.path.join(ABS_DIR_PATH, app_config['val_data_path'])
# test_data_abs_path = os.path.join(ABS_DIR_PATH, app_config['test_data_path'])


# train_df = pd.read_csv(train_data_abs_path)
# val_df = pd.read_csv(val_data_abs_path)
# test_df = pd.read_csv(test_data_abs_path)
#
# train_sentences = train_df["review"].to_numpy()
# train_labels = train_df["sentiment"].to_numpy()
#
# val_sentences = val_df["review"].to_numpy()
# val_labels = val_df["sentiment"].to_numpy()
#
# test_sentences = test_df["review"].to_numpy()
# test_labels = test_df["sentiment"].to_numpy()

# mp = ModelPrediction(test_sentences, test_labels)


def read_data(train_data_abs_path_func, val_data_abs_path_func, test_data_abs_path_func):
    train_df = pd.read_csv(train_data_abs_path_func)
    val_df = pd.read_csv(val_data_abs_path_func)
    test_df = pd.read_csv(test_data_abs_path_func)

    train_sentences = train_df["review"].to_numpy()
    train_labels = train_df["sentiment"].to_numpy()

    val_sentences = val_df["review"].to_numpy()
    val_labels = val_df["sentiment"].to_numpy()

    test_sentences = test_df["review"].to_numpy()
    test_labels = test_df["sentiment"].to_numpy()
    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels


def model_0_training(model_predictor, train_sent, train_lab):
    model_0 = Model0().model(train_sent, train_lab)
    baseline_results = model_predictor.pred(model_0, "naive_bayes")
    return model_0, baseline_results


def model_1_training(model_predictor, train_sent, train_lab, val_sent, val_lab):
    tf_model_1 = Model1(train_sent, train_lab, val_sent, val_lab).model()
    model_1_results = model_predictor.pred(tf_model_1, "simple_dense")
    return tf_model_1, model_1_results


def model_2_training(model_predictor, train_sent, train_lab, val_sent, val_lab):
    tf_model_2 = Model2(train_sent, train_lab, val_sent, val_lab).model()
    model_2_results = model_predictor.pred(tf_model_2, "lstm")
    return tf_model_2, model_2_results


def model_3_training(model_predictor, train_sent, train_lab, val_sent, val_lab):
    tf_model_3 = Model3(train_sent, train_lab, val_sent, val_lab).model()
    model_3_results = model_predictor.pred(tf_model_3, "gru")
    return tf_model_3, model_3_results


def model_4_training(model_predictor, train_sent, train_lab, val_sent, val_lab):
    tf_model_4 = Model4(train_sent, train_lab, val_sent, val_lab).model()
    model_4_results = model_predictor.pred(tf_model_4, "bidirectional")
    return tf_model_4, model_4_results


def model_5_training(model_predictor, train_sent, train_lab, val_sent, val_lab):
    tf_model_5 = Model5(train_sent, train_lab, val_sent, val_lab).model()
    model_5_results = model_predictor.pred(tf_model_5, "conv1d")
    return tf_model_5, model_5_results


def model_6_training(model_predictor, train_sent, train_lab, val_sent, val_lab):
    tf_model_6 = Model6(train_sent, train_lab, val_sent, val_lab).model()
    model_6_results = model_predictor.pred(tf_model_6, "tf_hub_sentence_encoder")
    return tf_model_6, model_6_results


def model_7_training(model_predictor, train_sent, train_lab, val_sent, val_lab):
    tf_model_7 = Model7(train_sent, train_lab, val_sent, val_lab).model()
    model_7_results = model_predictor.pred(tf_model_7, "tf_hub_10_percent_data")
    return model_7_results


if __name__ == '__main__':
    options = vars(opts)
    if options["data"] not in ("norm", "over", "under"):
        raise "wrong data sampling techniques"

    train_data_abs_path = os.path.join(ABS_DIR_PATH, app_config[f'train_{options["data"]}_sampled_data_path'])
    val_data_abs_path = os.path.join(ABS_DIR_PATH, app_config[f'val_{options["data"]}_sampled_data_path'])
    test_data_abs_path = os.path.join(ABS_DIR_PATH, app_config[f'test_{options["data"]}_sampled_data_path'])

    print(train_data_abs_path)
    print(train_data_abs_path)
    print(train_data_abs_path)

    train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels = read_data(
        train_data_abs_path, val_data_abs_path, test_data_abs_path)
    mp = ModelPrediction(test_sentences, test_labels)

    if options["models"] == "all":
        model_0, baseline_results = model_0_training(mp, train_sentences, train_labels)
        tf_model_1, model_1_results = model_1_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        tf_model_2, model_2_results = model_2_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        tf_model_3, model_3_results = model_3_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        tf_model_4, model_4_results = model_4_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        tf_model_5, model_5_results = model_5_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        tf_model_6, model_6_results = model_6_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        tf_model_7, model_7_results = model_7_training(mp, train_sentences, train_labels, val_sentences, val_labels)
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
        all_models_results_f1_score_path = ALL_MODELS_RESULT_PLOT_ABS_PATH.format(
            "all_models_f1_score_under_sampled.png")
        all_model_results_fig = all_model_results.sort_values("f1", ascending=False)["f1"].plot(kind="bar",
                                                                                                figsize=(10, 7))
        plt.savefig(all_models_results_f1_score_path, dpi=300)
        tf_model_1.save(model_1_abs_path)
        tf_model_2.save(model_2_abs_path)
        tf_model_3.save(model_3_abs_path)
        tf_model_4.save(model_4_abs_path)
        tf_model_5.save(model_5_abs_path)
        tf_model_6.save(model_6_abs_path)
        tf_model_7.save(model_7_abs_path)

    if options["models"] == "simple_dense":
        model_0, baseline_results = model_0_training(mp, train_sentences, train_labels)
        tf_model_1, model_1_results = model_1_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        print(compare_baseline_to_new_results(baseline_results, model_1_results))
        tf_model_1.save(model_1_abs_path)

    if options["models"] == "lstm":
        model_0, baseline_results = model_0_training(mp, train_sentences, train_labels)
        tf_model_2, model_2_results = model_2_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        print(compare_baseline_to_new_results(baseline_results, model_2_results))
        tf_model_2.save(model_1_abs_path)

    if options["models"] == "gru":
        model_0, baseline_results = model_0_training(mp, train_sentences, train_labels)
        tf_model_3, model_3_results = model_3_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        print(compare_baseline_to_new_results(baseline_results, model_3_results))
        tf_model_3.save(model_1_abs_path)

    if options["models"] == "bidirectional":
        model_0, baseline_results = model_0_training(mp, train_sentences, train_labels)
        tf_model_4, model_4_results = model_4_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        print(compare_baseline_to_new_results(baseline_results, model_4_results))
        tf_model_4.save(model_1_abs_path)

    if options["models"] == "conv1d":
        model_0, baseline_results = model_0_training(mp, train_sentences, train_labels)
        tf_model_5, model_5_results = model_5_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        print(compare_baseline_to_new_results(baseline_results, model_5_results))
        tf_model_5.save(model_1_abs_path)

    if options["models"] == "tf_hub_sentence_encoder":
        model_0, baseline_results = model_0_training(mp, train_sentences, train_labels)
        tf_model_6, model_6_results = model_6_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        print(compare_baseline_to_new_results(baseline_results, model_6_results))
        tf_model_6.save(model_1_abs_path)

    if options["models"] == "tf_hub_10_percent_data":
        model_0, baseline_results = model_0_training(mp, train_sentences, train_labels)
        tf_model_7, model_7_results = model_7_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        print(compare_baseline_to_new_results(baseline_results, model_7_results))
        tf_model_7.save(model_1_abs_path)

    if options["models"] == "ensemble":
        model_0, baseline_results = model_0_training(mp, train_sentences, train_labels)
        tf_model_2, model_2_results = model_2_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        tf_model_6, model_6_results = model_6_training(mp, train_sentences, train_labels, val_sentences, val_labels)
        baseline_pred_probs = np.max(model_0.predict_proba(val_sentences), axis=1)
        model_2_pred_probs = tf.squeeze(tf_model_2.predict(val_sentences), axis=1)
        model_6_pred_probs = tf.squeeze(tf_model_6.predict(val_sentences))
        combined_pred_probs = baseline_pred_probs + model_2_pred_probs + model_6_pred_probs
        combined_preds = tf.round(combined_pred_probs / 3)
        print(combined_preds[:20])
        ensemble_results = calculate_results(val_labels, combined_preds)
        print(ensemble_results)
