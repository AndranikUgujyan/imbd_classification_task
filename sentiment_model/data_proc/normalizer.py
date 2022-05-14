import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


# nltk.download('stopwords')
# nltk.download("maxent_treebank_pos_tagger")
# nltk.download("maxent_ne_chunker")
# nltk.download("punkt")
# nltk.download(["tagsets", "universal_tagset"])
# nltk.download('omw-1.4')

class TextNormalizer:

    def __init__(self):
        self._lemmatizer = WordNetLemmatizer()

        self._stop_words = set([sw for sw in stopwords.words('english') if sw != 'from'])

    def normalize(self, raw_text):
        refined_label = self.refine_label(raw_text)
        removed_standalone_numbers_text = re.sub(r'\b[0-9]+\b\s*', '', refined_label)
        words = word_tokenize(removed_standalone_numbers_text)
        processed_input = ' '.join(
            [self._lemmatizer.lemmatize(it.lower()) for it in words if it.lower() not in self._stop_words])
        return processed_input

    @staticmethod
    def refine_label(label: str):
        label = label.strip()
        label = label.lower()
        label = '_'.join(label.split(' '))
        return label
