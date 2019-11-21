import itertools
import os

import numpy as np
import scipy.special
import skimage.color
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_color_iter = itertools.cycle(
    ['navy', 'c', 'cornflowerblue', 'gold', 'darkorange']
)


def plot_results(x, y, means, covariances, title):
    splot = plt.subplot(1, 1, 1)
    if len(covariances.shape) == 1:
        covariances = covariances[:, np.newaxis, np.newaxis] * np.eye(x.shape[1], x.shape[1])
    if len(covariances.shape) == 2:
        covariances = covariances[:, :, np.newaxis] * np.eye(x.shape[1], x.shape[1])[np.newaxis, :, :]
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, _color_iter)):
        v, w = scipy.linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / scipy.linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(y == i):
            continue
        plt.scatter(x[y == i, 0], x[y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-10., 10.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks()
    plt.yticks()


def plot_samples(x, y, n_components, index, title):
    plt.subplot(5, 1, 4 + index)
    for i, color in zip(range(n_components), _color_iter):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(y == i):
            continue
        plt.scatter(x[y == i, 0], x[y == i, 1], .8, color=color)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


def load_data(path, cm='rgb'):
    labels_path = os.path.join(path, 'groundtruth')
    image_path = os.path.join(path, 'input')
    label_files = list(sorted(os.listdir(labels_path)))
    image_files = list(sorted(os.listdir(image_path)))

    labels, images = [], []
    for label_file, image_file in tqdm(zip(label_files, image_files), total=len(label_files)):
        label = skimage.color.rgb2gray(mpimg.imread(os.path.join(labels_path, label_file)))
        image = mpimg.imread(os.path.join(image_path, image_file))
        if cm == 'gray':
            image = skimage.color.rgb2gray(image) * 255
        if cm == 'hsv':
            image = skimage.color.rgb2hsv(image)
        labels.append(label)
        images.append(image)

    return np.array(labels), np.array(images, dtype=np.float)
