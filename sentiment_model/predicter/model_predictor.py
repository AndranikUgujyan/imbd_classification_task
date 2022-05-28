from sentiment_model import logger
from sentiment_model.data_proc.normalizer import TextNormalizer
from sentiment_model.saving_loading.model_saving_loading import ModelSavingLoading
import tensorflow as tf


class ModelPredictor:

    def __init__(self, model_path: str):
        loader = ModelSavingLoading()
        self._model = loader.load_model(model_path)
        self._normalizer = TextNormalizer()

    def predict(self, review):
        logger.debug('start prediction')
        logger.debug(f'start normalizing input texts: {review}')
        normalized_review = [self._normalizer.normalize(review)]
        logger.debug(f'normalized review: {normalized_review}')
        pred_prob = self._model.predict(normalized_review)
        pred_label = tf.squeeze(tf.round(pred_prob)).numpy()
        if 0 < pred_label:
            label = "positive"
        else:
            label = "negative"
        prediction_result = {"label": str(label), "probability": str(pred_prob[0][0])}
        return prediction_result
