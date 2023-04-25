from typing import Tuple, List, Union, Dict, Callable, Any, Optional

import os
import multiprocessing
import json
import numpy as np
import functools
import logging
from collections import Counter

from tqdm import tqdm

from s2and.data import ANDData
from s2and.consts import (
    CACHE_ROOT,
    NUMPY_NAN,
    FEATURIZER_VERSION,
    LARGE_INTEGER,
    DEFAULT_CHUNK_SIZE,
)
from s2and.text import (
    equal,
    equal_middle,
    diff,
    name_counts,
    TEXT_FUNCTIONS,
    name_text_features,
    jaccard,
    dice,
    counter_jaccard,
    counter_dice,
    cosine_sim,
    get_tfidf_cosine,
)

logger = logging.getLogger("s2and")

TupleOfArrays = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]

CACHED_FEATURES: Dict[str, Dict[str, Any]] = {}


class FeaturizationInfo:
    """
    Class to store information about how to generate and cache features

    Inputs:
        features_to_use: List[str[]]
            list of feature types to use
        featurizer_version: int
            What version of the featurizer we are on. This should be
            incremented when changing how features are computed so that a new cache
            is created
    """

    def __init__(
        self,
        features_to_use: List[str] = [
            "affiliation_similarity",
            "coauthor_similarity",
            "venue_similarity",
            "journal_similarity",
            "title_similarity",
            "reference_authors_similarity",
            "reference_titles_similarity",
            "reference_journals_similarity",
            "year_diff",
            "misc_features"
        ],
        featurizer_version: int = FEATURIZER_VERSION,
    ):
        self.features_to_use = features_to_use

        self.feature_group_to_index = {
            "affiliation_similarity": [0,1,2,3,4,5],
            "coauthor_similarity": [6,7,8,9,10],
            "venue_similarity": [11,12,13,14,15,16],
            "journal_similarity": [17,18,19,20,21,22],
            "title_similarity": [23,24,25,26,27,28,29],
            "reference_authors_similarity": [30,31,32,33],
            "reference_titles_similarity": [34,35,36,37],
            "reference_journals_similarity": [38,39,40,41],
            "year_diff": [42,43,44,45],
            "misc_features": [46,47,48,49,50,51]
        }

        self.number_of_features = max(functools.reduce(max, self.feature_group_to_index.values())) + 1  # type: ignore

        # NOTE: Increment this anytime a change is made to the featurization logic
        self.featurizer_version = featurizer_version

    def get_feature_names(self) -> List[str]:
        """
        Gets all of the feature names

        Returns
        -------
        List[string]: List of all the features names
        """
        feature_names = []

        # affiliation features
        if "affiliation_similarity" in self.features_to_use:
            feature_names.append("affiliation_char_gram_Jaccard")
            feature_names.append("affiliation_char_gram_Dice")
            feature_names.append("affiliation_word_gram_Jaccard")
            feature_names.append("affiliation_word_gram_Dice")
            feature_names.append("affiliation_word_tfidf_cosine")
            feature_names.append("affiliation_word_movers_distance")


        # co author features
        if "coauthor_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "coauthor_char_gram_Jaccard",
                    "coauthor_char_gram_Dice",
                    "coauthor_fullname_Jaccard",
                    "coauthor_fullname_Dice",
                    "coauthor_levenshtein"
                ]
            )

        # venue features
        if "venue_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "venue_char_gram_Jaccard",
                    "venue_char_gram_Dice",
                    "venue_word_gram_Jaccard",
                    "venue_word_gram_Dice",
                    "venue_word_tfidf_cosine",
                    "venue_word_movers_distance",
                ]
            )
        if "journal_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "journal_char_gram_Jaccard",
                    "journal_char_gram_Dice",
                    "journal_word_gram_Jaccard",
                    "journal_word_gram_Dice",
                    "journal_word_tfidf_cosine",
                    "journal_word_movers_distance",
                ]
            )
        # title features
        if "title_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "title_char_gram_Jaccard",
                    "title_char_gram_Dice",
                    "title_word_gram_Jaccard",
                    "title_word_gram_Dice",
                    "title_word_tfidf_cosine",
                    "title_word_movers_distance",
                    "title_embedding_cosine",
                ]
            )

        # reference features
        if "reference_authors_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "references_authors_char_gram_Jaccard",
                    "references_authors_char_gram_Dice",
                    "references_authors_word_gram_Jaccard",
                    "references_authors_word_gram_Dice",
                ]
            )

        if "reference_titles_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "references_titles_char_gram_Jaccard",
                    "references_titles_char_gram_Dice",
                    "references_titles_word_gram_Jaccard",
                    "references_titles_word_gram_Dice",
                ]
            )

        if "reference_journals_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "references_journals_char_gram_Jaccard",
                    "references_journals_char_gram_Dice",
                    "references_journals_word_gram_Jaccard",
                    "references_journals_word_gram_Dice",
                ]
            )

        # year features
        if "year_diff" in self.features_to_use:
            feature_names.extend(
                [
                    "year_diff_10",
                    "year_diff_15",
                    "year_diff_25",
                    "year_diff_50",                
                ])


        # position features
        if "misc_features" in self.features_to_use:
            feature_names.extend(
                ["self_citation","position_diff", "abstract_count", "english_count", "same_language", "language_reliability_count"]
            )

        return feature_names

    @staticmethod
    def feature_cache_key(signature_pair: Tuple) -> str:
        """
        returns the key in the feature cache dictionary for a signature pair

        Parameters
        ----------
        signature_pair: Tuple[string]
            pair of signature ids

        Returns
        -------
        string: the cache key
        """
        return signature_pair[0] + "___" + signature_pair[1]

    def cache_directory(self, dataset_name: str) -> str:
        """
        returns the cache directory for this dataset and featurizer version

        Parameters
        ----------
        dataset_name: string
            the name of the dataset

        Returns
        -------
        string: the cache directory
        """
        return os.path.join(CACHE_ROOT, dataset_name, str(self.featurizer_version))

    def cache_file_path(self, dataset_name: str) -> str:
        """
        returns the file path for the features cache

        Parameters
        ----------
        dataset_name: string
            the name of the dataset

        Returns
        -------
        string: the full file path for the features cache file
        """
        return os.path.join(
            self.cache_directory(dataset_name),
            "all_features.json",
        )

    def write_cache(self, cached_features: Dict, dataset_name: str):
        """
        Writes the cached features to the features cache file

        Parameters
        ----------
        cached_features: Dict
            the features, keyed by signature pair
        dataset_name: str
            the name of the dataset

        Returns
        -------
        nothing, writes the cache file
        """
        with open(
            self.cache_file_path(dataset_name),
            "w",
        ) as _json_file:
            json.dump(cached_features, _json_file)


NUM_FEATURES = FeaturizationInfo().number_of_features


def _single_pair_featurize(work_input: Tuple[str, str], index: int = -1) -> Tuple[List[Union[int, float]], int]:
    """
    Creates the features array for a single signature pair
    NOTE: This function uses a global variable to support faster multiprocessing. That means that this function
    should only be called from the many_pairs_featurize function below (or if you have carefully set your own global
    variable)

    Parameters
    ----------
    work_input: Tuple[str, str]
        pair of signature ids
    index: int
        the index of the pair in the list of all pairs,
        used to keep track of cached features

    Returns
    -------
    Tuple: tuple of the features array, and the index, which is simply passed through
    """
    global global_dataset

    features = []

    signature_1 = global_dataset.signatures[work_input[0]]  # type: ignore
    signature_2 = global_dataset.signatures[work_input[1]]  # type: ignore

    paper_id_1 = signature_1.paper_id
    paper_id_2 = signature_2.paper_id

    paper_1 = global_dataset.papers[str(paper_id_1)]  # type: ignore
    paper_2 = global_dataset.papers[str(paper_id_2)]  # type: ignore
    
    # affiliation
    features.extend(
        [
            counter_jaccard(
                signature_1.author_info_affiliations_ngrams_words,
                signature_2.author_info_affiliations_ngrams_words,
            ),
            counter_dice(
                signature_1.author_info_affiliations_ngrams_words,
                signature_2.author_info_affiliations_ngrams_words,
            ),
            counter_jaccard(
                signature_1.author_info_affiliations_ngrams_chars,
                signature_2.author_info_affiliations_ngrams_chars,
            ),
            counter_dice(
                signature_1.author_info_affiliations_ngrams_chars,
                signature_2.author_info_affiliations_ngrams_chars,
            ),
            get_tfidf_cosine(signature_1.author_info_affiliations_word_split, \
                 signature_2.author_info_affiliations_word_split, global_dataset.idf_counts['affiliation'])\
                    if (signature_1.author_info_affiliations_word_split and signature_2.author_info_affiliations_word_split) else NUMPY_NAN,
            global_dataset.W2Vmodels['affiliation'].wv.wmdistance(signature_1.author_info_affiliations_word_split,
                signature_2.author_info_affiliations_word_split) if (signature_1.author_info_affiliations_word_split \
                    and signature_2.author_info_affiliations_word_split) else NUMPY_NAN,
        ]
    )

    #coauthor
    features.extend(
        [
            # jaccard(signature_1.author_info_coauthor_blocks, signature_2.author_info_coauthor_blocks),
            counter_jaccard(
                signature_1.author_info_coauthor_ngrams_chars,
                signature_2.author_info_coauthor_ngrams_chars,
                denominator_max=5000,
            ),
            counter_dice(
                signature_1.author_info_coauthor_ngrams_chars,
                signature_2.author_info_coauthor_ngrams_chars,
                denominator_max=5000,
            ),
            jaccard(signature_1.author_info_coauthors, signature_2.author_info_coauthors),
            dice(signature_1.author_info_coauthors, signature_2.author_info_coauthors),
            TEXT_FUNCTIONS[0][0](" ".join(signature_1.author_info_coauthors)," ".join(signature_2.author_info_coauthors))\
                if (len(signature_1.author_info_coauthors) != 0 and len(signature_2.author_info_coauthors) != 0) else NUMPY_NAN,
        ]
    )

    #venue
    features.extend(
        [
            counter_jaccard(paper_1.venue_ngrams, paper_2.venue_ngrams),
            counter_dice(paper_1.venue_ngrams, paper_2.venue_ngrams),
            counter_jaccard(paper_1.venue_ngrams_words, paper_2.venue_ngrams_words),
            counter_dice(paper_1.venue_ngrams_words, paper_2.venue_ngrams_words),
            get_tfidf_cosine(paper_1.venue_word_split, paper_2.venue_word_split, global_dataset.idf_counts['venue'])\
                if (paper_1.venue_word_split and paper_2.venue_word_split) else NUMPY_NAN,
            global_dataset.W2Vmodels['venue'].wv.wmdistance(paper_1.venue_word_split, paper_2.venue_word_split) \
                    if (paper_1.venue_word_split and paper_2.venue_word_split) else NUMPY_NAN,            
        ]
    )

    #journal
    features.extend(
        [
            counter_jaccard(paper_1.journal_ngrams, paper_2.journal_ngrams),
            counter_dice(paper_1.venue_ngrams, paper_2.venue_ngrams),
            counter_jaccard(paper_1.journal_ngrams_words, paper_2.journal_ngrams_words),
            counter_dice(paper_1.journal_ngrams_words, paper_2.journal_ngrams_words),
            get_tfidf_cosine(paper_1.journal_word_split, paper_2.journal_word_split, global_dataset.idf_counts['venue'])\
                if (paper_1.journal_word_split and paper_2.journal_word_split) else NUMPY_NAN,
            global_dataset.W2Vmodels['venue'].wv.wmdistance(paper_1.journal_word_split, paper_2.journal_word_split) \
                    if (paper_1.journal_word_split and paper_2.journal_word_split) else NUMPY_NAN,            
        ]
        )

    # title   
    english_or_unknown_count = int(paper_1.predicted_language in {"en", "un"}) + int(
        paper_2.predicted_language in {"en", "un"}
    )

    specter_1 = None
    specter_2 = None
    if english_or_unknown_count == 2 and global_dataset.specter_embeddings is not None:  # type: ignore
        if str(paper_id_1) in global_dataset.specter_embeddings:  # type: ignore
            specter_1 = global_dataset.specter_embeddings[str(paper_id_1)]  # type: ignore
            if np.all(specter_1 == 0):
                specter_1 = None
        if str(paper_id_2) in global_dataset.specter_embeddings:  # type: ignore
            specter_2 = global_dataset.specter_embeddings[str(paper_id_2)]  # type: ignore
            if np.all(specter_2 == 0):
                specter_2 = None

    features.extend(
        [
            counter_jaccard(paper_1.title_ngrams_chars, paper_2.title_ngrams_chars),  
            counter_dice(paper_1.title_ngrams_chars, paper_2.title_ngrams_chars),
            counter_jaccard(paper_1.title_ngrams_words, paper_2.title_ngrams_words),
            counter_dice(paper_1.title_ngrams_words, paper_2.title_ngrams_words),                     
            get_tfidf_cosine(paper_1.title_word_split, paper_2.title_word_split, global_dataset.idf_counts['title'])\
                if (paper_1.title_word_split and paper_2.title_word_split)  else NUMPY_NAN,
            global_dataset.W2Vmodels['title'].wv.wmdistance(paper_1.title_word_split, paper_2.title_word_split) \
                if (paper_1.title_word_split and paper_2.title_word_split)  else NUMPY_NAN,
            cosine_sim(specter_1, specter_2) + 1 if (specter_1 is not None and specter_2 is not None) else NUMPY_NAN,
        ]
    )

    #reference_author
    features.extend(
        [
            counter_jaccard(paper_1.reference_details[0][0], paper_2.reference_details[0][0], denominator_max=5000),
            counter_dice(paper_1.reference_details[0][0], paper_2.reference_details[0][0], denominator_max=5000),
            counter_jaccard(paper_1.reference_details[0][1], paper_2.reference_details[0][1], denominator_max=5000),
            counter_dice(paper_1.reference_details[0][1], paper_2.reference_details[0][1], denominator_max=5000),
        ]
    )
    #reference_title
    features.extend(
        [    
            counter_jaccard(paper_1.reference_details[1][0], paper_2.reference_details[1][0]),
            counter_dice(paper_1.reference_details[1][0], paper_2.reference_details[1][0]),
            counter_jaccard(paper_1.reference_details[1][1], paper_2.reference_details[1][1]),
            counter_dice(paper_1.reference_details[1][1], paper_2.reference_details[1][1]),
        ]
    )
    #reference_journal
    features.extend(
        [    
            counter_jaccard(paper_1.reference_details[2][0], paper_2.reference_details[2][0]),
            counter_dice(paper_1.reference_details[2][0], paper_2.reference_details[2][0]),
            counter_jaccard(paper_1.reference_details[2][1], paper_2.reference_details[2][1]),
            counter_dice(paper_1.reference_details[2][1], paper_2.reference_details[2][1]),
        ]
    )
    # counter_jaccard(paper_1.reference_details[3], paper_2.reference_details[3]), blocks

    #year
    year_diff = diff(paper_1.year if paper_1.year is not None and paper_1.year > 0 else None,
        paper_2.year if paper_2.year is not None and paper_2.year > 0 else None,)
    features.extend(
        [
        np.minimum(
            year_diff,
            10,
        ),
        np.minimum(
            year_diff,
            15,
        ),
        np.minimum(
            year_diff,
            25,
        ),
        np.minimum(
            year_diff,
            50,
        )]
    )  # 10, 15, 25, 50

    #misc
    references_1 = set(paper_1.references)
    references_2 = set(paper_2.references)
    features.extend(
        [
            int(paper_id_2 in references_1 or paper_id_1 in references_2),
            np.minimum(
                diff(
                    signature_1.author_info_position,
                    signature_2.author_info_position,
                ),
                50,
            ),
            int(paper_1.has_abstract) + int(paper_2.has_abstract),
            english_or_unknown_count,
            paper_1.predicted_language == paper_2.predicted_language,
            int(paper_1.is_reliable) + int(paper_2.is_reliable),
        ]
    )

    # unifying feature type in features array
    features = [float(val) if type(val) in [np.float32, np.float64, float] else int(val) for val in features]

    return features, index


def parallel_helper(piece_of_work: Tuple, worker_func: Callable):
    """
    Helper function to explode tuple arguments

    Parameters
    ----------
    piece_of_work: Tuple
        the input for the worker func, in tuple form
    worker_func: Callable
        the function that will do the work

    Returns
    -------
    returns the result of calling the worker function
    """
    result = worker_func(*piece_of_work)
    return result


def many_pairs_featurize(
    signature_pairs: List[Tuple[str, str, Union[int, float]]],
    dataset: ANDData,
    featurizer_info: FeaturizationInfo,
    n_jobs: int,
    use_cache: bool,
    chunk_size: int,
    nameless_featurizer_info: Optional[FeaturizationInfo] = None,
    nan_value: float = np.nan,
    delete_training_data: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Featurizes many pairs

    Parameters
    ----------
    signature_pairs: List[pairs]
        the pairs to featurize
    dataset: ANDData
        the dataset containing the relevant data
    featurizer_info: FeaturizationInfo
        the FeautrizationInfo object containing the listing of features to use
        and featurizer version
    n_jobs: int
        the number of cpus to use
    use_cache: bool
        whether or not to use write to/read from the features cache
    chunk_size: int
        the chunk size for multiprocessing
    nameless_featurizer_info: FeaturizationInfo
        the FeaturizationInfo for creating the features that do not use any name features,
        these will not be computed if this is None
    nan_value: float
        the value to replace nans with
    delete_training_data: bool
        Whether to delete some suspicious training rows

    Returns
    -------
    np.ndarray: the main features for all the pairs
    np.ndarray: the labels for all the pairs
    np.ndarray: the nameless features for all the pairs
    """
    global global_dataset
    global_dataset = dataset  # type: ignore

    cached_features: Dict[str, Any] = {"features": {}}
    cache_changed = False
    if use_cache:
        logger.info("Loading cache...")
        if not os.path.exists(featurizer_info.cache_directory(dataset.name)):
            os.makedirs(featurizer_info.cache_directory(dataset.name))
        if os.path.exists(featurizer_info.cache_file_path(dataset.name)):
            if featurizer_info.cache_file_path(dataset.name) in CACHED_FEATURES:
                cached_features = CACHED_FEATURES[featurizer_info.cache_file_path(dataset.name)]
            else:
                with open(featurizer_info.cache_file_path(dataset.name)) as _json_file:
                    cached_features = json.load(_json_file)
                logger.info(f"Cache loaded with {len(cached_features['features'])} keys")
        else:
            logger.info("Cache initiated.")
            cached_features = {}
            cached_features["features"] = {}
            cached_features["features_to_use"] = featurizer_info.features_to_use

    features = np.ones((len(signature_pairs), NUM_FEATURES)) * (-LARGE_INTEGER)
    labels = np.zeros(len(signature_pairs))
    pieces_of_work = []
    logger.info(f"Creating {len(signature_pairs)} pieces of work")
    for i, pair in tqdm(enumerate(signature_pairs), desc="Creating work", disable=len(signature_pairs) <= 100000):
        labels[i] = pair[2]

        # negative labels are an indication of partial supervision
        if pair[2] < 0:
            continue

        cache_key = pair[0] + "___" + pair[1]
        if use_cache and cache_key in cached_features["features"]:
            cached_vector = cached_features["features"][cache_key]
            features[i, :] = cached_vector
            continue

        cache_key = pair[1] + "___" + pair[0]
        if use_cache and cache_key in cached_features["features"]:
            cached_vector = cached_features["features"][cache_key]
            features[i, :] = cached_vector
            continue

        cache_changed = True
        pieces_of_work.append(((pair[0], pair[1]), i))

    logger.info("Created pieces of work")

    indices_to_use = set()
    for feature_name in featurizer_info.features_to_use:
        indices_to_use.update(featurizer_info.feature_group_to_index[feature_name])
    indices_to_use: List[int] = sorted(list(indices_to_use))  # type: ignore

    if nameless_featurizer_info:
        nameless_indices_to_use = set()
        for feature_name in nameless_featurizer_info.features_to_use:
            nameless_indices_to_use.update(nameless_featurizer_info.feature_group_to_index[feature_name])
        nameless_indices_to_use: List[int] = sorted(list(nameless_indices_to_use))  # type: ignore

    if cache_changed:
        if n_jobs > 1:
            logger.info(f"Cached changed, doing {len(pieces_of_work)} work in parallel")
            with multiprocessing.Pool(processes=n_jobs if len(pieces_of_work) > 1000 else 1) as p:
                _max = len(pieces_of_work)
                with tqdm(total=_max, desc="Doing work", disable=_max <= 10000) as pbar:
                    for (feature_output, index) in p.imap(
                        functools.partial(parallel_helper, worker_func=_single_pair_featurize),
                        pieces_of_work,
                        min(chunk_size, max(1, int((_max / n_jobs) / 2))),
                    ):
                        # Write to in memory cache if we're not skipping
                        if use_cache:
                            cached_features["features"][
                                featurizer_info.feature_cache_key(signature_pairs[index])
                            ] = feature_output
                        features[index, :] = feature_output
                        pbar.update()
        else:
            logger.info(f"Cached changed, doing {len(pieces_of_work)} work in serial")
            partial_func = functools.partial(parallel_helper, worker_func=_single_pair_featurize)
            for piece in tqdm(pieces_of_work, total=len(pieces_of_work), desc="Doing work"):
                result = partial_func(piece)
                if use_cache:
                    cached_features["features"][featurizer_info.feature_cache_key(signature_pairs[result[1]])] = result[
                        0
                    ]
                print(result[0])
                features[result[1], :] = result[0]
        logger.info("Work completed")

    if use_cache and cache_changed:
        logger.info("Writing to on disk cache")
        featurizer_info.write_cache(cached_features, dataset.name)
        logger.info(f"Cache written with {len(cached_features['features'])} keys.")

    if use_cache:
        logger.info("Writing to in memory cache")
        CACHED_FEATURES[featurizer_info.cache_file_path(dataset.name)] = cached_features
        logger.info("In memory cache written")

    if delete_training_data:
        logger.info("Deleting some training rows")
        negative_label_indices = labels == 0
        high_coauthor_sim_indices = features[:, featurizer_info.get_feature_names().index("coauthor_similarity")] > 0.95
        indices_to_remove = negative_label_indices & high_coauthor_sim_indices
        logger.info(f"Intending to remove {sum(indices_to_remove)} rows")
        original_size = len(labels)
        features = features[~indices_to_remove, :]
        labels = labels[~indices_to_remove]
        logger.info(f"Removed {original_size - features.shape[0]} rows and {original_size - len(labels)} labels")

    logger.info("Making numpy arrays for features and labels")
    # have to do this before subselecting features
    if nameless_featurizer_info is not None:
        nameless_features = features[:, nameless_indices_to_use]
        nameless_features[np.isnan(nameless_features)] = nan_value
    else:
        nameless_features = None

    features = features[:, indices_to_use]
    features[np.isnan(features)] = nan_value

    logger.info("Numpy arrays made")
    return features, labels, nameless_features


def featurize(
    dataset: ANDData,
    featurizer_info: FeaturizationInfo,
    n_jobs: int = 1,
    use_cache: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nameless_featurizer_info: Optional[FeaturizationInfo] = None,
    nan_value: float = np.nan,
    delete_training_data: bool = False,
    all_pair: bool = False,
) -> Union[Tuple[TupleOfArrays, TupleOfArrays, TupleOfArrays], TupleOfArrays]:
    """
    Featurizes the input dataset

    Parameters
    ----------
    dataset: ANDData
        the dataset containing the relevant data
    featurizer_info: FeaturizationInfo
        the FeautrizationInfo object containing the listing of features to use
        and featurizer version
    n_jobs: int
        the number of cpus to use
    use_cache: bool
        whether or not to use write to/read from the features cache
    chunk_size: int
        the chunk size for multiprocessing
    nameless_featurizer_info: FeaturizationInfo
        the FeaturizationInfo for creating the features that do not use any name features,
        these will not be computed if this is None
    nan_value: float
        the value to replace nans with
    delete_training_data: bool
        Whether to delete some suspicious training examples

    Returns
    -------
    train/val/test features and labels if mode is 'train',
    features and labels for all pairs if mode is 'inference'
    """
    if dataset.mode == "inference":
        logger.info("featurizing all pairs")
        all_pairs = dataset.all_pairs()
        all_features = many_pairs_featurize(
            all_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
        )
        logger.info("featurized all pairs")
        return all_features
    else:
        if dataset.train_pairs is None:
            if dataset.train_blocks is not None:
                (
                    train_signatures,
                    val_signatures,
                    test_signatures,
                ) = dataset.split_cluster_signatures_fixed()
            elif dataset.train_signatures is not None:
                (
                    train_signatures,
                    val_signatures,
                    test_signatures,
                ) = dataset.split_data_signatures_fixed()
            else:
                (
                    train_signatures,
                    val_signatures,
                    test_signatures,
                ) = dataset.split_cluster_signatures()  # type: ignore

            train_pairs, val_pairs, test_pairs = dataset.split_pairs(train_signatures, val_signatures, test_signatures, all_pair)

        else:
            train_pairs, val_pairs, test_pairs = dataset.fixed_pairs()

        logger.info("featurizing train")
        train_features = many_pairs_featurize(
            train_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            delete_training_data,
        )
        logger.info("featurized train, featurizing val")
        val_features = many_pairs_featurize(
            val_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
        )
        logger.info("featurized val, featurizing test")
        test_features = many_pairs_featurize(
            test_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
        )
        logger.info("featurized test")
        return train_features, val_features, test_features

def preprocess_features(features, phase='train', params=None):
    def cal_nan_meanpercls(x, y):
        mean_p = []
        mean_n = []
        mean = []
        for i in range(x.shape[1]):
            notnanrows = ~np.isnan(x[:,i])   
            X_nonnan = x[notnanrows,i]
            y_nonnan = y[notnanrows]
            X_nonnan_p = X_nonnan[y_nonnan==1]
            X_nonnan_n = X_nonnan[y_nonnan==0]
            mean.append(np.mean(X_nonnan)) 
            mean_p.append(np.mean(X_nonnan_p))
            mean_n.append(np.mean(X_nonnan_n))
        return mean, mean_p, mean_n

    def fill_nan_meanpercls(x, y, phase, params):
        for i in range(x.shape[1]):
            if phase == 'train':
                notnanrows = ~np.isnan(x[:,i])    
                x[np.logical_and(~notnanrows,y==1),i] = np.mean(params['mean_p'][i])
                x[np.logical_and(~notnanrows,y==0),i] = np.mean(params['mean_n'][i])
            elif phase == 'test':
                nanrows = np.isnan(x[:,i])    
                x[nanrows,i] = np.mean(params['mean'][i])
        return x

    def normalize(x, params):
        x = (x - params['min']) / ( params['max'] - params['min'])
        x[:,np.argwhere((params['max'] - params['min'])==0).reshape(-1)] = 1
        # attr_df[:,distance_list] = 1 - attr_df[:,distance_list]
        return x

    if params is None:
        params = {}
        params['mean'], params['mean_p'], params['mean_n'] = cal_nan_meanpercls(features[0], features[1])
    X_pre = fill_nan_meanpercls(features[0], features[1], phase, params)
    if 'min' not in params:
        params['min'], params['max'] = np.min(X_pre,axis=0), np.max(X_pre,axis=0)
    X_pre = normalize(X_pre, params)
    return X_pre, features[1], features[2], params