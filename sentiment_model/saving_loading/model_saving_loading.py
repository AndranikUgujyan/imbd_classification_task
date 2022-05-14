import tensorflow as tf
import tensorflow_addons as tfa
from sentiment_model import logger
from sentiment_model.utils.help_func import error_response


class ModelSavingLoading:

    def __init__(self):
        super(ModelSavingLoading, self).__init__()

    def load_model(self, model_path):
        logger.debug(f'Start loading model from {model_path}')
        try:

            model = tf.keras.models.load_model(model_path,
                                                   custom_objects={"loss": tfa.losses.SigmoidFocalCrossEntropy()})
            logger.debug(f'Loading complete from {model_path}')
            return model
        except Exception as err:
            return error_response(f'error={err}', logger)

    def save_model(self, model_for_save, model_path):
        logger.debug(f'Start saving model {model_for_save}')
        try:
            model_for_save.save(model_path)
            return True
        except Exception as err:
            return error_response(f'error={err}', logger)


# if __name__ == "__main__":
#     model_6_path = "/home/andranik/dev/imbd_classification/models/model_5"
#
#     ms = ModelSavingLoading()
#     model_t = ms.load_model(model_6_path)
#     def predict_on_sentence(model, sentence):
#         """
#         Uses model to make a prediction on sentence.
#
#         Returns the sentence, the predicted label and the prediction probability.
#         """
#         pred_prob = model.predict([sentence])
#         print(pred_prob)
#         pred_label = tf.squeeze(tf.round(pred_prob)).numpy()
#         print(f"Pred: {pred_label}", "(real disaster)" if pred_label > 0 else "(not real disaster)",
#               f"Prob: {pred_prob[0][0]}")
#         print(f"Text:\n{sentence}")
#
#
#     r = {"review": "A very good story for a film which if done prop"
#                    "erly would be quite interesting, but where the he"
#                    "ll is the ending to this film?<br /><br />In fact, "
#                    "what is the point of it?<br /><br />The scenes zip "
#                    "through so quick that you felt you were not part of"
#                    " the film emotionally, and the feeling of being detac"
#                    "hed from understanding the storyline.<br /><br />The p"
#                    "erformances of the cast are questionable, if not believable"
#                    ".<br /><br />Did I miss the conclusion somewhere in the film?"
#                    " I guess we have to wait for the sequel.<br /><br />"}
#     predict_on_sentence(model=model_t,  # use the USE model
#                         sentence=r["review"])