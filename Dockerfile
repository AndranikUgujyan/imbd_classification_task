FROM python:3.9

ENV APP_HOME /app
WORKDIR $APP_HOME

ADD requirements.txt .
# Install production dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader 'punkt' -d /usr/local/nltk_data
RUN python -m nltk.downloader 'wordnet' -d /usr/local/nltk_data
RUN python -m nltk.downloader 'stopwords' -d /usr/local/nltk_data
RUN python -m nltk.downloader 'maxent_treebank_pos_tagger' -d /usr/local/nltk_data
RUN python -m nltk.downloader 'wordnet' -d /usr/local/nltk_data

ADD sentiment_model  sentiment_model
ADD configs configs
ADD sentiment_model/models models

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 --max-requests-jitter 10 'sentiment_model.app:app'