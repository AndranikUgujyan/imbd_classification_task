from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime
import tensorflow as tf
import yaml


def get_input_text(data, log):
    try:
        log.debug(f'started')
        if data["review"]:
            return data["review"]
    except Exception as e:
        log.exception(f"can't get input text from: {data}")
        raise ValueError(str(e))


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.
    :param y_true:  true labels in the form of a 1D array
    :param y_pred: predicted labels in the form of a 1D array
    :return: dictionary of accuracy, precision, recall, f1-score
    """
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results


def compare_baseline_to_new_results(baseline_results, new_model_results):
    for key, value in baseline_results.items():
        result = f"Baseline {key}: {value:.2f}, New {key}: {new_model_results[key]:.2f}, " \
                 f"Difference: {new_model_results[key] - value:.2f}"
        return result


def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instand to store log files and stores log files with the filepath.
    :param dir_name: target directory to store TensorBoard log files
    :param experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    :return:
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def error_response(message, log):
    log.debug(f"started, message={message}")
    return {"result": "Fail", "ErrorMessage": message}, 500


def load_cfg(yaml_file_path: str):
    """
    Load a YAML configuration file.
    """

    with open(yaml_file_path, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.SafeLoader)
    return cfg
