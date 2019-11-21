import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


def highlight_mask(image, mask):
    """
    :param image: numpy.array(M x N x 3) [0, 255]
    :param mask: numpy.array(M x N, dtype=np.bool)
    :return: image with highlighted mask
    """

    m, n = mask.shape

    im = image.copy()
    mask_vec = mask.ravel()

    im_vec = im.reshape((m * n, 3))
    im_vec[mask_vec, :] /= 2.0
    im_vec[mask_vec, 0] += 128

    return im


def make_video(frames, result, get_mask=None):
    """
    Play video using frames with highlighted masks
    :param frames: numpy.array(D x M x N x 3) [0, 255]
    :param result: numpy.array(D x M x N, dtype=np.bool)
    :param get_mask:
    D - number of frames.

    Example of usage:
    video = make_video(test_img, ans)
    video()
    """

    highlighted_frames = np.zeros(frames.shape)
    for i in range(frames.shape[0]):
        mask = result[i, :, :]
        if get_mask is not None:
            mask = get_mask(mask)
        highlighted_frames[i, :, :, :] = highlight_mask(frames[i, :, :, :], mask)

    fig = plt.figure()
    im = plt.imshow(highlighted_frames[0, :, :, :].astype(np.uint8))

    def updatefig(j):
        im.set_array(highlighted_frames[j, :, :, :].astype(np.uint8))
        return im,

    return lambda: animation.FuncAnimation(fig, updatefig, frames=highlighted_frames.shape[0], interval=100, blit=True)
