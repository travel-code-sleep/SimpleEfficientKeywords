from typing import Optional, Union, Iterable
import gc
import warnings
import pandas as pd
import nltk
import textacy
from textacy import preprocessing
import textacy.extract as ext
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from unidecode import unidecode

from tqdm import tqdm
tqdm.pandas()

warnings.filterwarnings("ignore")

nlp_lg = spacy.load('en_core_web_lg')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
stopwords = set(stopwords + list(STOP_WORDS))
stopwords = {w.lower() for w in stopwords}


class SEKeywords:
    """ _summary_

    _extended_summary_
    """

    def __init__(self) -> None:
        pass

    def preprocess_text(self, text, clean_text, dedupe_text,):
        pass

    def _select_clean_level(self, level):
        pass

    def _fraction_keywords(self, n_words):
        pass

    def _create_spacy_doc(self, text):
        pass

    def sentiment(self, model):
        pass

    def _select_part_of_speech(self, pos_def):
        pass

    def _textrank_keywords(self, doc):
        pass

    def _sgrank_keywords(self, doc):
        pass

    def _clean_dedupe_keywords(self, keywords):
        pass

    def _extract_noun_chunks(self, doc):
        pass

    def _extract_proper_nouns(self, doc):
        pass

    def _extract_entities(self, doc):
        pass

    def extract_keywords(
            self, data, text_col, is_doc,
            include_part_of_speech,
            include_noun_chunks,
            include_entities,
            clean_text,
            dedupe_text,
            lemmatize,) -> pd.DataFrame:
        pass

    def aggregate_keywords(self, keywords, topics):
        pass


class SEKeywordsAggregate:
    """ _summary_

    _extended_summary_
    """
    pass
