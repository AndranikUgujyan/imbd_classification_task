from sentiment_model import logger
from sentiment_model.data_proc.normalizer import TextNormalizer
from sentiment_model.saving_loading.model_saving_loading import ModelSavingLoading
import tensorflow as tf


class ModelPredictor:
    __INSTANCE = None

    def __init__(self, model_path: str):
        if ModelPredictor.__INSTANCE is None:
            loader = ModelSavingLoading()
            self._model = loader.load_model(model_path)
            self._normalizer = TextNormalizer()
            ModelPredictor.__INSTANCE = self

    @classmethod
    def get_instance(cls, model_path=None):
        if cls.__INSTANCE is None:
            cls.__INSTANCE = ModelPredictor(model_path)
        return cls.__INSTANCE

    @classmethod
    def clear_instance(cls):
        cls.__INSTANCE = None

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


# if __name__ == "__main__":
#     model_6_path = "/home/andranik/dev/imbd_classification/models/model_5"
#
#     mp = ModelPredictor(model_6_path)
#     r = {
#         "review": "A very good story for a film which if done properly would be quite interesting, but where the hell is the ending to this film?<br /><br />In fact, what is the point of it?<br /><br />The scenes zip through so quick that you felt you were not part of the film emotionally, and the feeling of being detached from understanding the storyline.<br /><br />The performances of the cast are questionable, if not believable.<br /><br />Did I miss the conclusion somewhere in the film? I guess we have to wait for the sequel.<br /><br />"}
#     print(mp.predict(r["review"]))
