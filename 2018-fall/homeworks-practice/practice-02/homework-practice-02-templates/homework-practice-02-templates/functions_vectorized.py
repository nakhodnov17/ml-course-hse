import numpy as np


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """
    diag = np.diag(x)
    non_zero_diag = diag[diag != 0]
    if non_zero_diag.shape[0] == 0:
        return None
    return np.prod(non_zero_diag)


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """
    return np.all(np.sort(x) == np.sort(y))


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """
    cnt_zeros = x.shape[0] - np.count_nonzero(x)
    if cnt_zeros == 0 or cnt_zeros == 1 and x[-1] == 0:
        return None
    idx_after_zero = np.where(np.concatenate([[1], x]) == 0)[0]
    idx_after_zero[idx_after_zero > x.shape[0] - 1] = idx_after_zero[0]
    return np.max(x[idx_after_zero])


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x num_channels)
    coefs -- 1-d numpy array (length num_channels)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """
    return np.dot(img, coefs)


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    x_nan = np.concatenate([[np.nan], x, [np.nan]])

    arr_1 = x[~(x_nan[1:] == np.roll(x_nan[1:], -1))[:-1]]
    arr_2 = (np.argwhere(~(x_nan[1:] == np.roll(x_nan[1:], -1))[:-1]).reshape(-1) -
             np.argwhere(~(x_nan[:-1] == np.roll(x_nan[:-1], 1))[1:]).reshape(-1) + 1)

    return arr_1, arr_2


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vectorized implementation.
    """
    return np.sqrt(
        np.sum(x ** 2, axis=1)[:, np.newaxis] +
        np.sum(y ** 2, axis=1) +
        -2. * np.dot(x, y.T)
    )
