import os
import sentiment_model
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from sentiment_model import app_config
from sentiment_model.utils.help_func import calculate_results

abs_dir_path = os.path.dirname(os.path.abspath(sentiment_model.__file__))

CM_SAV_PATH = os.path.join(abs_dir_path, app_config['conf_matrix_save_path'])


class ModelPrediction:

    def __init__(self, data_for_predict, true_data):
        self.data_for_predict = data_for_predict
        self.true_data = true_data

    def pred(self, model, model_name):
        model_pred_probs = model.predict(self.data_for_predict)
        model_preds = tf.squeeze(tf.round(model_pred_probs))
        model_results = calculate_results(self.true_data, model_preds)
        cr = classification_report(y_true=self.true_data,
                                   y_pred=model_preds,
                                   zero_division=0)
        print(model_name)
        print(cr)
        cm = confusion_matrix(self.true_data, model_preds, labels=[0, 1])
        cmd = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
        cmd.plot().figure_.savefig(CM_SAV_PATH.format(model_name))
        return model_results
