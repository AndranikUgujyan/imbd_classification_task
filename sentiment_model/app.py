import os
import json
import os
import sentiment_model
from flask import Flask, request

from sentiment_model import app_config, logger
from sentiment_model.predicter.model_predictor import ModelPredictor
from sentiment_model.utils.help_func import error_response, get_input_text

abs_dir_path = os.path.dirname(os.path.abspath(sentiment_model.__file__))

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str)

OPTIONS = vars(parser.parse_args())

def init_app():
    app = Flask(__name__)
    if OPTIONS["model"] == "simple_dense":
        model_path = os.path.join(abs_dir_path, app_config['model_1_path'])
        ModelPredictor.get_instance(model_path)
    if OPTIONS["model"] == "lstm":
        model_path = os.path.join(abs_dir_path, app_config['model_2_path'])
        ModelPredictor.get_instance(model_path)
    if OPTIONS["model"] == "gru":
        model_path = os.path.join(abs_dir_path, app_config['model_3_path'])
        ModelPredictor.get_instance(model_path)
    if OPTIONS["model"] == "bidirectional":
        model_path = os.path.join(abs_dir_path, app_config['model_4_path'])
        ModelPredictor.get_instance(model_path)
    if OPTIONS["model"] == "conv1d":
        model_path = os.path.join(abs_dir_path, app_config['model_5_path'])
        ModelPredictor.get_instance(model_path)
    if OPTIONS["model"] == "tf_hub_sentence_encoder":
        model_path = os.path.join(abs_dir_path, app_config['model_6_path'])
        ModelPredictor.get_instance(model_path)
    if OPTIONS["model"] == "tf_hub_10_percent_data":
        model_path = os.path.join(abs_dir_path, app_config['model_7_path'])
        ModelPredictor.get_instance(model_path)

    return app


app = init_app()


@app.route('/identify_sentiment', methods=['POST'])
def predict_sentiment():
    try:
        log_headers_and_ip(request)
        _data = request.data.decode('utf-8')
        logger.debug(f'started, data:{_data}')
        j_obj = json.loads(_data)
        logger.debug('get fasttext client instance')
        model_client = ModelPredictor.get_instance()
        texts = get_input_text(j_obj, logger)
        prediction = model_client.predict(texts)
        response = app.response_class(response=json.dumps([prediction], indent=True),
                                      status=200,
                                      mimetype='application/json')
        return response
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
