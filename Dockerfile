FROM python:3.9

ADD requirements.txt .
# Install production dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install nltk downloader.
RUN python -m nltk.downloader 'punkt' -d /usr/local/lib/nltk_data
RUN python -m nltk.downloader 'wordnet' -d /usr/local/lib/nltk_data
RUN python -m nltk.downloader 'stopwords' -d /usr/local/lib/nltk_data
RUN python -m nltk.downloader 'maxent_treebank_pos_tagger' -d /usr/local/lib/nltk_data
RUN python -m nltk.downloader 'omw-1.4' -d /usr/local/lib/nltk_data


ADD sentiment_model sentiment_model
ADD configs configs

#ENV PORT 8080

RUN python3 -m sentiment_model.app
#ENV PORT 8080
#
#CMD [ "python3", "-m" , "sentiment_model.app"]
