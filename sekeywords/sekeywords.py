"""sekeywords is the wrapper module for keywords extractor and aggregator."""
from typing import Optional, Callable, Union, Iterable
import gc
import warnings
import pandas as pd
import nltk
# import textacy
# from textacy import preprocessing
import textacy.extract as ext
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

    def _preprocess_text(
            self, data: pd.DataFrame, text_col: str,
            convert_to_ascii: bool, clean_text: bool,
            dedupe_text: bool, dedupe_subset: tuple,
            cleaning_func: Callable) -> pd.DataFrame:
        if not cleaning_func:
            data[text_col] = data[text_col].str.lower(
            ) if clean_text else data[text_col]
            data[text_col] = data[text_col].map(
                unidecode) if convert_to_ascii else data[text_col]
        else:
            data[text_col] = data[text_col].map(cleaning_func)

        data = data.drop_duplicates(subset=list(
            dedupe_subset+tuple(text_col))) if dedupe_text else data

        return data

    def _create_spacy_doc(self, texts: list, n_process: int, batch_size: int):
        text_doc = list(nlp_lg.pipe(texts, n_process=n_process,
                        batch_size=batch_size))
        del texts
        gc.collect()
        return text_doc

    def _get_n_keywords(self, n_words: int, keywords_frac: float) -> float:
        if keywords_frac:
            n_keywords = keywords_frac * n_words

        n_keywords = 0.06 * n_words if n_words <= 1000 else 0.012 * n_words
        return round(n_keywords)

    def _extract_textrank_keywords(
            self, doc, normalize: str, n_keywords_to_extract: int,
            include_part_of_speech: tuple, textrank_window_size: int) -> list:
        return [
            i[0] for i in ext.keyterms.textrank(
                doc, window_size=textrank_window_size, normalize=normalize, topn=n_keywords_to_extract,
                include_pos=include_part_of_speech)
            if i[1] > 0.02
        ]

    def _extract_sgrank_keywords(
            self, doc, normalize: str, n_keywords_to_extract: int,
            include_part_of_speech: tuple, sgrank_ngrams_range: tuple) -> list:
        return [
            i[0] for i in ext.keyterms.sgrank(
                doc, ngrams=sgrank_ngrams_range,
                normalize=normalize, topn=n_keywords_to_extract, include_pos=include_part_of_speech)
            if i[1] > 0.02
        ]

    def _extract_noun_chunks(self, doc) -> tuple[list, list, list]:
        noun_chunks = [
            nc for nc in doc.noun_chunks if nc.text.lower() not in stopwords
        ]
        return [nc.text.lower() for nc in noun_chunks if len(nc.text.split()) == 1]

    def _clean_dedupe_keywords(self, keywords, single_noun_chunks) -> tuple[list, list]:
        keywords = sorted(
            list(set(keywords)),
            reverse=False,
            key=lambda x: len(x)
        )
        bad_words = []
        if len(keywords) > 0:
            for i in range(len(keywords)):
                bad_words.extend(
                    keywords[i]
                    for j in range(i + 1, len(keywords))
                    if keywords[i] in keywords[j]
                )
        if single_noun_chunks:
            bad_words.extend(i for i in single_noun_chunks if i in keywords)
        return bad_words

    def _extract_entities(self, doc, entity_types: tuple):
        return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in entity_types]

    def _run_extraction_job(self, doc, normalize: str, keywords_frac: float,
                            include_part_of_speech: tuple, textrank_window_size: int,
                            sgrank_ngrams_range: tuple,
                            handle_noun_chunks: bool) -> tuple[list, list]:

        word_count = len(doc.text.split())

        n_keywords_to_extract = self._get_n_keywords(
            word_count, keywords_frac) if word_count > 10 else 1

        textrank_keywords = self._extract_textrank_keywords(
            doc, normalize, n_keywords_to_extract, include_part_of_speech, textrank_window_size)

        sgrank_keywords = self._extract_sgrank_keywords(
            doc, normalize, n_keywords_to_extract, include_part_of_speech, sgrank_ngrams_range)

        keywords = textrank_keywords + sgrank_keywords

        if handle_noun_chunks:
            single_noun_chunks = self._extract_noun_chunks(
                doc)
        single_noun_chunks = None

        bad_words = self._clean_dedupe_keywords(keywords, single_noun_chunks)

        keywords = list({word for word in keywords if word not in bad_words})

        return keywords, bad_words

    def extract_keywords(
            self, data: Union[pd.DataFrame, Iterable],
            text_col: Optional[str] = None,
            convert_to_ascii: bool = True,
            clean_text: bool = True,
            dedupe_text: bool = False,
            dedupe_subset: Optional[tuple] = None,
            lemmatize: bool = True,
            include_part_of_speech: tuple = ('NOUN', 'PROPN', 'ADJ'),
            # is_doc: bool = False,
            cleaning_func: Optional[Callable] = None,
            n_process: int = 4,
            batch_size: int = 1000,
            keywords_frac: Optional[float] = None,
            textrank_window_size: int = 8,
            sgrank_ngrams_range: tuple = (2, 3, 4, 5, 6, 7, 8),
            handle_noun_chunks: bool = False,
            include_entities: bool = False,
            entity_types: tuple = ('ORG', 'PRODUCT', 'GPE')) -> Union[pd.DataFrame, tuple[list, list]]:
        """extract_keywords _summary_

        _extended_summary_

        Args:
            data (Union[pd.DataFrame, Iterable]): _description_
            text_col (Optional[str], optional): _description_. Defaults to None.
            convert_to_ascii (bool, optional): _description_. Defaults to True.
            clean_text (bool, optional): _description_. Defaults to True.
            dedupe_text (bool, optional): _description_. Defaults to False.
            dedupe_subset (Optional[tuple], optional): _description_. Defaults to None.
            lemmatize (bool, optional): _description_. Defaults to True.
            include_part_of_speech (tuple, optional): _description_. Defaults to ('NOUN', 'PROPN', 'ADJ').
            cleaning_func (Optional[Callable], optional): _description_. Defaults to None.
            n_process (int, optional): _description_. Defaults to 4.
            batch_size (int, optional): _description_. Defaults to 1000.
            keywords_frac (Optional[float], optional): _description_. Defaults to None.
            textrank_window_size (int, optional): _description_. Defaults to 8.
            sgrank_ngrams_range (tuple, optional): _description_. Defaults to (2, 3, 4, 5, 6, 7, 8).
            handle_noun_chunks (bool, optional): _description_. Defaults to False.
            include_entities (bool, optional): _description_. Defaults to False.
            entity_types (tuple, optional): _description_. Defaults to ('ORG', 'PRODUCT', 'GPE').

        Returns:
            Union[pd.DataFrame, tuple[list, list]]: _description_
        """

        # if not is_doc:
        if isinstance(data, pd.DataFrame):
            data = data.copy(deep=True)
            dataframe_op = True
        else:
            text_col = 'text'
            data = pd.DataFrame(data, columns=[text_col])
            dataframe_op = False

        data = self._preprocess_text(
            data, text_col, convert_to_ascii, clean_text, dedupe_text, dedupe_subset, cleaning_func)

        data['text_spacy_doc'] = self._create_spacy_doc(
            data[text_col].tolist(), n_process, batch_size)

        normalize = 'lemma' if lemmatize else 'lower'

        data['keywords'], data['bad_words'] = zip(*data.apply(
            lambda x: self._run_extraction_job(
                x.text_spacy_doc,
                normalize=normalize,
                keywords_frac=keywords_frac,
                include_part_of_speech=include_part_of_speech,
                textrank_window_size=textrank_window_size,
                sgrank_ngrams_range=sgrank_ngrams_range,
                handle_noun_chunks=handle_noun_chunks,
            ),
            axis=1))

        if include_entities:
            data['entities'], data['entity_types'] = zip(*data.text_spacy_doc.apply(
                lambda x: self._extract_entities(x, entity_types)))
        data.drop(columns='text_spacy_doc', inplace=True)

        if dataframe_op:
            return data

        return data.keywords.tolist(), data.bad_words.tolist()


class SEKeywordsAggregate:
    """ _summary_

    _extended_summary_
    """
    pass
    # def aggregate_keywords(self, data, keywords, topics, dedupe_text):
    #     pass
    # #     if dedupe_text:
    # #         if topic_col:
    # #             topic_col = topic_col if isinstance(
    # #                 topic_col, tuple) else tuple(topic_col)
    # #         topic_col = topic_col + tuple(text_col)
    # #         data = data.drop_duplicates(subset=list(topic_col))
    # #     return data

    # def sentiment(self, model):
    #     pass
