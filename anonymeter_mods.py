from typing import List, Optional, Union, Tuple, Dict

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from numba import float64, int64, jit
from math import fabs, isnan
import pprint

pp = pprint.PrettyPrinter(indent=4)
debug = False

'''
The code in here was cut-n-paste from the github repo anonymeter, with minor additions.
'''

@jit(nopython=True, nogil=True)
def gower_distance(r0: np.ndarray, r1: np.ndarray, cat_cols_index: np.ndarray) -> float64:
    r"""Distance between two records inspired by the Gower distance [1].

    To handle mixed type data, the distance is specialized for numerical (continuous)
    and categorical data. For numerical records, we use the L1 norm,
    computed after the columns have been normalized so that :math:`d(a_i, b_i)\leq 1`
    for every :math:`a_i`, :math:`b_i`. For categorical, :math:`d(a_i, b_i)` is 1,
    if the entries :math:`a_i`, :math:`b_i` differ, else, it is 0.

    Notes
    -----
    To keep the balance between numerical and categorical values, the input records
    have to be properly normalized. Their numerical part need to be scaled so that
    the difference between any two values of a column (from both dataset) is *at most* 1.

    References
    ----------
    [1]. `Gower (1971) "A general coefficient of similarity and some of its properties.
    <https://www.jstor.org/stable/2528823?seq=1>`_

    Parameters
    ----------
    r0 : np.array
        Input array of shape (D,).
    r1 : np.array
        Input array of shape (D,).
    cat_cols_index : int
        Index delimiting the categorical columns in r0/r1 if present. For example,
        ``r0[:cat_cols_index]`` are the numerical columns, and ``r0[cat_cols_index:]`` are
        the categorical ones. For a fully numerical dataset, use ``cat_cols_index =
        len(r0)``. For a fully categorical one, set ``cat_cols_index`` to 0.

    Returns
    -------
    float
        distance between the records.

    """
    dist = 0.0

    for i in range(len(r0)):
        if isnan(r0[i]) and isnan(r1[i]):
            dist += 1
        else:
            if i < cat_cols_index:
                dist += fabs(r0[i] - r1[i])
            else:
                if r0[i] != r1[i]:
                    dist += 1
    return dist

@jit(nopython=True, nogil=True)
def _nearest_neighbors(queries, candidates, cat_cols_index, n_neighbors):
    r"""For every element of ``queries``, find its nearest neighbors in ``candidates``.

    Parameters
    ----------
    queries : np.ndarray
        Input array of shape (Nx, D).
    candidates : np.ndarray
        Input array of shape (Ny, D).
    n_neighbors : int
        Determines the number of closest neighbors per entry to be returned.
    cat_cols_idx : int
        Index delimiting the categorical columns in X/Y, if present.

    Returns
    -------
    idx : np.ndarray[int]
        Array of shape (Nx, n_neighbors). For each element in ``queries``,
        this array contains the indices of the closest neighbors in
        ``candidates``. That is, ``candidates[idx[i]]`` are the elements of
        ``candidates`` that are closer to ``queries[i]``.
    lps : np.ndarray[float]
        Array of shape (Nx, n_neighbors). This array containing the distances
        between the record pairs identified by idx.

    """
    idx = np.zeros((queries.shape[0], n_neighbors), dtype=int64)
    dists = np.zeros((queries.shape[0], n_neighbors), dtype=float64)

    for ix in range(queries.shape[0]):

        dist_ix = np.zeros((candidates.shape[0]), dtype=float64)

        for iy in range(candidates.shape[0]):

            dist_ix[iy] = gower_distance(r0=queries[ix], r1=candidates[iy], cat_cols_index=cat_cols_index)

        close_match_idx = dist_ix.argsort()[:n_neighbors]
        idx[ix] = close_match_idx
        dists[ix] = dist_ix[close_match_idx]

    return idx, dists


def _encode_categorical(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode dataframes with categorical values keeping label consistend."""
    encoded = pd.concat((df1, df2), keys=["df1", "df2"])

    for col in encoded.columns:
        encoded[col] = LabelEncoder().fit_transform(encoded[col])

    return encoded.loc["df1"], encoded.loc["df2"]


def _scale_numerical(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scale dataframes with *only* numerical values."""
    df1_min, df1_max = df1.min(), df1.max()
    df2_min, df2_max = df2.min(), df2.max()

    mins = df1_min.where(df1_min < df2_min, df2_min)
    maxs = df1_max.where(df1_max > df2_max, df2_max)
    ranges = maxs - mins

    if any(ranges == 0):
        cnames = ", ".join(ranges[ranges == 0].index.values)
        ranges[ranges == 0] = 1

    df1_scaled = df1.apply(lambda x: x / ranges[x.name])
    df2_scaled = df2.apply(lambda x: x / ranges[x.name])

    return df1_scaled, df2_scaled

def mixed_types_transform(
    df1: pd.DataFrame, df2: pd.DataFrame, num_cols: List[str], cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combination of an encoder and a scaler to treat mixed type data.

    Numerical columns are scaled by dividing them by their range across both
    datasets, so that the difference between any two values within a column will
    be smaller than or equal to one:
    x -> x' = x /  max{max(x), max(x_other)} - min{min(x), min(x_other)}

    Categorical columns are label encoded. This encoding is based on the
    `statice.preprocessing.encoders.DataframeEncoder` fitted on the firts
    dataframe, and applied to both of them.

    Parameters
    ----------
    df1: pd.DataFrame.
        Input DataFrame. This dataframe will be used to fit the DataframeLabelEncoder.
    df2: pd.DataFrame.
        Second input DataFrame.
    num_cols: list[str].
        Names of the numerical columns to be processed.
    cat_cols: list[str].
        Names of the  columns to be processed.

    Returns
    -------
    trans_df1: pd.DataFrame.
        Transformed df1.
    trans_df2: pd.DataFrame.
        Transformed df2.

    """
    if not set(df1.columns) == set(df2.columns):
        raise ValueError(f"Input dataframes have different columns. df1: {df1.columns}, df2: {df2.columns}.")

    if not set(num_cols + cat_cols) == set(df1.columns):
        raise ValueError(
            f"Dataframes columns {df1.columns} do not match "
            "with `num_cols` and `cat_cols`.\n"
            f"num_cols: {num_cols}\n"
            f"cat_cols: {cat_cols}"
        )

    df1_num, df2_num = pd.DataFrame(), pd.DataFrame()
    if len(num_cols) > 0:
        df1_num, df2_num = _scale_numerical(df1[num_cols], df2[num_cols])

    df1_cat, df2_cat = pd.DataFrame(), pd.DataFrame()
    if len(cat_cols) > 0:
        df1_cat, df2_cat = _encode_categorical(df1[cat_cols], df2[cat_cols])

    df1_out = pd.concat([df1_num, df1_cat], axis=1)[df1.columns]

    df2_out = pd.concat([df2_num, df2_cat], axis=1)[df2.columns]
    return df1_out, df2_out

def detect_col_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify numerical and non-numerical columns in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    Dict[str: List[str]]
        Dictionary with column names separated by types. Key of the dictionary are
        'num' or 'cat' (numerical and non-numerical, that is categorical, resp.).
        Values are lists of column names.

    """
    num_cols: List[str] = list(df.select_dtypes("number").columns.values)
    cat_cols: List[str] = [cn for cn in df.columns.values if cn not in num_cols]

    return {"num": sorted(num_cols), "cat": sorted(cat_cols)}

def detect_consistent_col_types(df1: pd.DataFrame, df2: pd.DataFrame):
    """Detect colum types for a pair dataframe an check that they are the same.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Input dataframe
    df2 : pandas.DataFrame
        Input dataframe

    Returns
    -------
    Dict[str: List[str]]
        Dictionary with column names separated by types. Key of the dictionary are
        'num' or 'cat' (numerical and non-numerical, that is categorical, resp.).
        Values are lists of column names.

    """
    ctypes1 = detect_col_types(df1)

    if ctypes1 != detect_col_types(df2):
        print(ctypes1)
        print(detect_col_types(df2))
        raise RuntimeError("Input dataframes have different column names/types.")

    return ctypes1

def get_matches(basis: pd.DataFrame, guess_idx: np.ndarray, aux_cols: list) -> pd.DataFrame:
    # Get the matching row
    try:
        match_row = basis.iloc[guess_idx[0]].squeeze()
    except Exception as e:
        print("basis.iloc exception occurred:", e)

    # Create a boolean mask where each row is True if the row matches the matching row for all aux_cols
    try:
        mask = (basis[aux_cols] == match_row[aux_cols]).all(axis=1)
    except Exception as e:
        print("mask exception occurred:", e)

    # Use the mask to get the matching rows from basis
    matching_rows_df = basis[mask]

    return matching_rows_df, match_row

def run_anonymeter_attack(
    targets: pd.DataFrame,    # This is the victim information
    basis: pd.DataFrame,      # This is control or synthetic data, depending
    aux_cols: List[str],
    secret: str,
    regression: Optional[bool],
) -> int:
    n_jobs = -2
    if regression is None:
        #regression = pd.api.types.is_numeric_dtype(target[secret])
        print(f"Not expecting regression is None")
        exit(1)

    nn = MixedTypeKNeighbors(n_jobs=n_jobs, n_neighbors=1).fit(candidates=basis[aux_cols])

    guess_idx = nn.kneighbors(queries=targets[aux_cols])
    guess = basis.iloc[guess_idx.flatten()][secret]
    # df_matching contains all of the rows that match the known attributes (aux_cols) of the best
    # matching row in the basis data
    df_matching, match_row = get_matches(basis=basis, guess_idx=guess_idx, aux_cols=aux_cols)
    match_modal_value = df_matching[secret].mode()[0]
    match_modal_count = (df_matching[secret] == match_modal_value).sum()
    percentage = 100*(match_modal_count / len(df_matching))
    if debug:
        distinct_values = df_matching[secret].nunique()
        print(f"got {len(df_matching)} matching rows")
        print(f"secret distinct values: {distinct_values}"	)
        print(f"match_modal_value: {match_modal_value}, match_modal_count: {match_modal_count}, percentage: {percentage}")
        if percentage >= 50:
            print("hit_50")
        if percentage >= 90:
            print("hit_90")
    
    ans = {'guess_series': guess,
            'match_row': match_row,
            'match_modal_value': match_modal_value,
            'match_modal_count': match_modal_count,
            'match_modal_percentage': percentage}
    if len(df_matching) == 0:
        print(f"Error: no matching rows")
        pp.pprint(ans)
        sys.exit(1)
    return ans
    #return evaluate_inference_guesses(guesses=guesses, secrets=targets[secret], regression=regression).sum()


def evaluate_inference_guesses(
    guesses: pd.Series, secrets: pd.Series, regression: bool, tolerance: float = 0.05
) -> np.ndarray:
    """Evaluate the success of an inference attack.

    The attack is successful if the attacker managed to make a correct guess.

    In case of regression problems, when the secret is a continuous variable,
    the guess is correct if the relative difference between guess and target
    is smaller than a given tolerance. In the case of categorical target
    variables, the inference is correct if the secrets are guessed exactly.

    Parameters
    ----------
    guesses : pd.Series
        Attacker guesses for each of the targets.
    secrets : pd.Series
        Array with the true values of the secret for each of the targets.
    regression : bool
        Whether or not the attacker is trying to solve a classification or
        a regression task. The first case is suitable for categorical or
        discrete secrets, the second for numerical continuous ones.
    tolerance : float, default is 0.05
        Maximum value for the relative difference between target and secret
        for the inference to be considered correct.

    Returns
    -------
    np.array
        Array of boolean values indicating the correcteness of each guess.

    """
    guesses = guesses.values
    secrets = secrets.values

    if regression:
        rel_abs_diff = np.abs(guesses - secrets) / (guesses + 1e-12)
        value_match = rel_abs_diff <= tolerance
    else:
        value_match = guesses == secrets

    nan_match = np.logical_and(pd.isnull(guesses), pd.isnull(secrets))

    return np.logical_or(nan_match, value_match)

class MixedTypeKNeighbors:
    """Nearest neighbor algorithm for mixed type data.

    To handle mixed type data, we use a distance function inspired by the Gower similarity.
    The distance is specialized for numerical (continuous) and categorical data. For
    numerical records, we use the L1 norm, computed after the columns have been
    normalized so that :math:`d(a_i, b_i) <= 1` for every :math:`a_i`, :math:`b_i`.
    For categorical, :math:`d(a_i, b_i)` is 1, if the entries :math:`a_i`, :math:`b_i`
    differ, else, it is 0.

    References
    ----------
    [1]. `Gower (1971) "A general coefficient of similarity and some of its properties.
    <https://www.jstor.org/stable/2528823?seq=1>`_

    Parameters
    ----------
    n_neighbors : int, default is 5
        Determines the number of closest neighbors per entry to be returned.
    n_jobs : int, default is -2
        Number of jobs to use. It follows joblib convention, so that ``n_jobs = -1``
        means all available cores.

    """

    def __init__(self, n_neighbors: int = 5, n_jobs: int = -2):
        self._n_neighbors = n_neighbors
        self._n_jobs = n_jobs

    def fit(self, candidates: pd.DataFrame, ctypes: Optional[Dict[str, List[str]]] = None):
        """Prepare for nearest neighbor search.

        Parameters
        ----------
        candidates : pd.DataFrame
            Dataset containing the records one would find the neighbors in.
        ctypes : dict, optional.
            Dictionary specifying which columns in X should be treated as
            continuous and which should be treated as categorical. For example,
            ``ctypes = {'num': ['distance'], 'cat': ['color']}`` specify the types
            of a two column dataset.

        """
        self._candidates = candidates
        self._ctypes = ctypes
        return self

    def kneighbors(
        self, queries: pd.DataFrame, n_neighbors: Optional[int] = None, return_distance: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Find the nearest neighbors for a set of query points.

        Note
        ----
        The search is performed in a brute-force fashion. For large datasets
        or large number of query points, the search for nearest neighbor will
        become very slow.

        Parameters
        ----------
        queries : pd.DataFrame
            Query points for the nearest neighbor searches.
        n_neighbors : int, default is None
            Number of neighbors required for each sample.
            The default is the value passed to the constructor.
        return_distance : bool, default is False
            Whether or not to return the distances of the neigbors or
            just the indexes.

        Returns
        -------
        np.narray of shape (df.shape[0], n_neighbors)
            Array with the indexes of the elements of the fit dataset closer to
            each element in the query dataset.
        np.narray of shape (df.shape[0], n_neighbors)
            Array with the distances of the neighbors pairs. This is optional and
            it is returned only if ``return_distances`` is ``True``

        """
        if n_neighbors is None:
            n_neighbors = self._n_neighbors

        if n_neighbors > self._candidates.shape[0]:
            n_neighbors = self._candidates.shape[0]

        if self._ctypes is None:
            self._ctypes = detect_consistent_col_types(df1=self._candidates, df2=queries)
        candidates, queries = mixed_types_transform(
            df1=self._candidates, df2=queries, num_cols=self._ctypes["num"], cat_cols=self._ctypes["cat"]
        )

        cols = self._ctypes["num"] + self._ctypes["cat"]
        queries = queries[cols].values
        candidates = candidates[cols].values

        with Parallel(n_jobs=self._n_jobs, backend="threading") as executor:

            res = executor(
                delayed(_nearest_neighbors)(
                    queries=queries[ii : ii + 1],
                    candidates=candidates,
                    cat_cols_index=len(self._ctypes["num"]),
                    n_neighbors=n_neighbors,
                )
                for ii in range(queries.shape[0])
            )

            indexes, distances = zip(*res)
            indexes, distances = np.vstack(indexes), np.vstack(distances)

        if return_distance:
            return distances, indexes

        return indexes
