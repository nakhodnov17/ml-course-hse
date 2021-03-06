import math


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    result = 1.
    for idx in range(min(len(x), len(x[0]))):
        if abs(x[idx][idx]) != 0:
            result *= x[idx][idx]
    return result


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    x_sorted = sorted(x)
    y_sorted = sorted(y)
    for idx in range(len(x)):
        if x_sorted[idx] != y_sorted[idx]:
            return False
    return True


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    result = -1e100
    for idx in range(len(x) - 1):
        if x[idx] == 0 and x[idx + 1] > result:
            result = x[idx + 1]
    return result


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x num_channels)
    coefs -- 1-d numpy array (length num_channels)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """
    result = []
    for idx in range(len(img)):
        line_result = []
        for jdx in range(len(img[idx])):
            pixel_result = 0.
            for zdx in range(len(img[idx][jdx])):
                pixel_result += img[idx][jdx][zdx] * coefs[zdx]
            line_result.append(pixel_result)
        result.append(line_result)
    return result


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    arr_1, arr_2 = [], []
    tmp_len = 1
    for idx in range(len(x) - 1):
        if x[idx] == x[idx + 1]:
            tmp_len += 1
        else:
            arr_1.append(x[idx])
            arr_2.append(tmp_len)
            tmp_len = 1
    arr_1.append(x[-1])
    arr_2.append(tmp_len)
    return arr_1, arr_2


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """
    result = []
    for x_value in x:
        line_result = []
        for y_value in y:
            distance = 0.
            for idx in range(len(x_value)):
                distance += (x_value[idx] - y_value[idx]) ** 2
            line_result.append(math.sqrt(distance))
        result.append(line_result)
    return result
