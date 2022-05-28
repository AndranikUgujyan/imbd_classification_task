import os
import json
import os
import sentiment_model
from flask import Flask, request

from sentiment_model import app_config, logger
from sentiment_model.predicter.model_predictor import ModelPredictor
from sentiment_model.utils.help_func import error_response, get_input_text

ABS_DIR_PATH = os.path.dirname(os.path.abspath(sentiment_model.__file__))

app = Flask(__name__)


@app.route('/identify_sentiment', methods=['POST'])
def predict_sentiment():
    try:
        log_headers_and_ip(request)
        _data = request.data.decode('utf-8')
        logger.debug(f'started, data:{_data}')
        j_obj = json.loads(_data)
        logger.debug('get client instance')
        model_path = config_model(j_obj["model"])
        model_client = ModelPredictor(model_path)
        texts = get_input_text(j_obj, logger)
        prediction = model_client.predict(texts)
        response = app.response_class(response=json.dumps([prediction], indent=True),
                                      status=200,
                                      mimetype='application/json')
        return response
    except Exception as err:
        return error_response(f'error={err}', logger)


def config_model(model_name):
    try:
        if model_name == "simple_dense":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_1_path'])
            return model_path
        if model_name == "lstm":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_2_path'])
            return model_path
        if model_name == "gru":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_3_path'])
            return model_path
        if model_name == "bidirectional":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_4_path'])
            return model_path
        if model_name == "conv1d":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_5_path'])
            return model_path
        if model_name == "tf_hub_sentence_encoder":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_6_path'])
            return model_path
        if model_name == "tf_hub_10_percent_data":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_7_path'])
            return model_path
    except Exception as err:
        return error_response(f'error={err}', logger)


def log_headers_and_ip(request):
    logger.debug('started')
    try:
        logger.debug(f'IP:{request.remote_addr}')
        logger.debug(f'headers:{request.headers}')
    except Exception:
        logger.exception('unable to log headers and IP.')


@app.errorhandler(500)
def server_error(e):
    logger.exception('error occurred during a request.')
    return f"An internal error occurred: <pre>{e}</pre>See logs for full stacktrace.", 500


if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 8080)))
