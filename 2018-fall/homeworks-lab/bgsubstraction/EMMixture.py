import numpy as np
import scipy.special
from tqdm import tqdm

from collections import namedtuple


class GaussianMixture:
    def __init__(
            self, n_components=1, covariance_type='full',
            reg_covar=1e-6, init_params='random'
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.init_params = init_params

    def get_parameters(self, x):
        parameters = namedtuple('Parameters', 'weights means covariances')
        if self.init_params == 'random':
            parameters.weights = np.random.random([self.n_components])
            parameters.weights /= np.sum(parameters.weights)
            parameters.means = np.random.uniform(
                np.percentile(x, 5, axis=0), np.percentile(x, 95, axis=0),
                [self.n_components, x.shape[1]]
            )
            if self.covariance_type == 'spherical':
                parameters.covariances = np.random.random([self.n_components])
            if self.covariance_type == 'tied':
                a = np.random.random([x.shape[1], x.shape[1]])
                parameters.covariances = (
                    np.matmul(a.T, a) +
                    self.reg_covar * np.eye(x.shape[1], x.shape[1])
                )
            if self.covariance_type == 'diag':
                parameters.covariances = np.random.random([self.n_components, x.shape[1]]) + self.reg_covar
            if self.covariance_type == 'full':
                a = np.random.random([self.n_components, x.shape[1], x.shape[1]])
                parameters.covariances = (
                    np.matmul(a.transpose((0, 2, 1)), a) +
                    self.reg_covar * np.eye(x.shape[1], x.shape[1])[np.newaxis, :, :]
                )
        return parameters

    def _estimate_covariances_spherical(self, x, responsibilities, weights_unnormed, means):
        return np.mean(self._estimate_covariances_diag(x, responsibilities, weights_unnormed, means), axis=1)

    def _estimate_covariances_tied(self, x, responsibilities, weights_unnormed, means):
        raise NotImplementedError

    def _estimate_covariances_diag(self, x, responsibilities, weights_unnormed, means):
        means_squared = means ** 2
        x_squared = np.dot(responsibilities.T, x * x) / weights_unnormed[:, np.newaxis]
        return x_squared - means_squared + self.reg_covar

    def _estimate_covariances_full(self, x, responsibilities, weights_unnormed, means):
        covariances = np.empty([means.shape[0], x.shape[1], x.shape[1]])
        for k in range(covariances.shape[0]):
            difference = x - means[k]
            covariances[k] = np.matmul(responsibilities[:, k] * difference.T, difference)
            covariances[k] /= weights_unnormed[k]
            covariances[k].flat[::x.shape[1] + 1] += self.reg_covar
        return covariances

    def estimate_parameters(self, x, responsibilities):
        parameters = namedtuple('Parameters', 'weights means covariances')

        weights_unnormed = np.sum(responsibilities, axis=0) + self.reg_covar
        means = np.matmul(responsibilities.T, x) / weights_unnormed[:, np.newaxis]
        covariances = {
            "spherical": self._estimate_covariances_spherical,
            "tied": self._estimate_covariances_tied,
            "diag": self._estimate_covariances_diag,
            "full": self._estimate_covariances_full
        }[self.covariance_type](x, responsibilities, weights_unnormed, means)

        (parameters.weights, parameters.means, parameters.covariances) \
            = (weights_unnormed / x.shape[0], means, covariances)
        return parameters

    # noinspection PyMethodMayBeStatic
    def _logunweightpdf_spherical(self, x, parameters):
        difference = x[:, np.newaxis, :] - parameters.means[np.newaxis, :, :]
        logunnormedunweightpdf = (
                - 1. / 2. / parameters.covariances[np.newaxis, :] * np.sum(difference * difference, axis=2)
                - x.shape[1] / 2. * np.log(2. * np.pi)
                - x.shape[1] / 2. * np.log(parameters.covariances)[np.newaxis, :]
        )
        return logunnormedunweightpdf

    def _logunweightpdf_tied(self, x, parameters):
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def _logunweightpdf_diag(self, x, parameters):
        difference = x[:, np.newaxis, :] - parameters.means[np.newaxis, :, :]
        logunnormedunweightpdf = (
            - 1. / 2. * np.sum(difference * difference / parameters.covariances[np.newaxis, :, :], axis=2)
            - x.shape[1] / 2. * np.log(2. * np.pi)
            - 1. / 2. * np.sum(np.log(parameters.covariances), axis=1)[np.newaxis, :]
        )
        return logunnormedunweightpdf

    def _logunweightpdf_full(self, x, parameters):
        logunweightpdf = np.empty([x.shape[0], self.n_components])
        for k in range(self.n_components):
            difference = x - parameters.means[k]
            logunweightpdf[:, k] = np.sum(
                np.matmul(
                    difference, np.linalg.inv(parameters.covariances[k])
                ) * difference,
                axis=1
            )
            logunweightpdf[:, k] = (
                - 1. / 2. * logunweightpdf[:, k]
                - x.shape[1] / 2. * np.log(2. * np.pi)
                - 1. / 2. * np.log(np.linalg.det(parameters.covariances[k]))
            )
        return logunweightpdf

    def logpdf(self, x, parameters) -> np.ndarray:
        logpdf = {
            "spherical": self._logunweightpdf_spherical,
            "tied": self._logunweightpdf_tied,
            "diag": self._logunweightpdf_diag,
            "full": self._logunweightpdf_full
        }[self.covariance_type](x, parameters) + np.log(parameters.weights)[np.newaxis, :]
        return logpdf

    # noinspection PyMethodMayBeStatic
    def _distance_spherical(self, x, parameters):
        difference = x[:, np.newaxis, :] - parameters.means[np.newaxis, :, :]
        distances = np.sqrt(np.sum(difference * difference, axis=2) / parameters.covariances[np.newaxis, :])
        return distances

    def _distance_tied(self, x, parameters):
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def _distance_diag(self, x, parameters):
        difference = x[:, np.newaxis, :] - parameters.means[np.newaxis, :, :]
        distances = np.sqrt(np.sum(difference * difference / parameters.covariances[np.newaxis, :, :], axis=2))
        return distances

    def _distance_full(self, x, parameters):
        distances = np.empty([x.shape[0], self.n_components])
        for k in range(self.n_components):
            difference = x - parameters.means[k]
            distances[:, k] = np.sqrt(
                np.sum(
                    np.matmul(
                        difference, np.linalg.inv(parameters.covariances[k])
                    ) * difference, axis=1
                )
            )
        return distances

    def distance(self, x, parameters):
        """
            Distance from point to distribution
            In case of Gaussian Mixture distance computes as Mahalanobis distance
        :param x:
        :param parameters:
        :return:
        """
        distances = {
            "spherical": self._distance_spherical,
            "tied": self._distance_tied,
            "diag": self._distance_diag,
            "full": self._distance_full
        }[self.covariance_type](x, parameters)
        return np.sum(distances * parameters.weights[np.newaxis, :], axis=1)

    def update(self, x, parameters, alpha):
        if x.size != 1 or self.n_components != 1:
            raise NotImplementedError
        difference = x - parameters.means
        parameters.covariances = (
                alpha * difference ** 2
                + (1. - alpha) * parameters.covariances
            ) + self.reg_covar
        parameters.means = alpha * x + (1. - alpha) * parameters.means
        return parameters

    # noinspection PyMethodMayBeStatic
    def sample(self, parameters):
        # result = np.empty([self.n_components, parameters.means.shape[1]])
        # for k in range(self.n_components):
        #     if self.covariance_type == 'full':
        #         result[k] = np.random.multivariate_normal(parameters.means[k], parameters.covariances[k])
        #     else:
        #         result[k] = np.random.normal(parameters.means[k], parameters.covariances[k], 1)
        # return np.matmul(parameters.weights, result)
        return np.matmul(parameters.weights, parameters.means)


class EM:
    def __init__(
            self, base_distribution=GaussianMixture, tol=1e-3,
            max_iter=100, n_init=1, verbose=False, save_history=False,
            *distribution_args, **distribution_kwargs
    ):
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.save_history = save_history
        self.max_iter = max_iter

        self.base_distribution = base_distribution(*distribution_args, **distribution_kwargs)

        self.parameters_ = None
        self.converged_ = False
        self.n_iter_ = None
        self.lower_bound_ = -np.inf
        self.history = []

    def _e_step(self, x, parameters):
        log_prob = self.base_distribution.logpdf(x, parameters)
        log_prob_norm = scipy.special.logsumexp(log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = log_prob - log_prob_norm[:, np.newaxis]
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        parameters = self.base_distribution.estimate_parameters(x, np.exp(log_resp))
        return parameters

    def fit(self, x):
        for n_init in range(self.n_init):
            current_init_n_iter = -1
            current_init_lower_bound = -np.inf
            init_history = []
            parameters = self.base_distribution.get_parameters(x)
            for n_iter in range(self.max_iter):
                current_iter_lower_bound, log_resp = self._e_step(x, parameters)
                parameters = self._m_step(x, log_resp)
                if self.save_history:
                    init_history.append(current_iter_lower_bound)
                if current_iter_lower_bound - current_init_lower_bound < self.tol:
                    self.converged_ = True
                    current_init_n_iter = n_iter
                    self.parameters_ = parameters
                    current_init_lower_bound = current_iter_lower_bound
                    break
                current_init_lower_bound = current_iter_lower_bound
                if self.verbose:
                    print("N_init: {0:d}, N_iter: {1:d}, LB: {2:.4f}".format(n_init, n_iter, current_init_lower_bound))
            if current_init_lower_bound > self.lower_bound_:
                self.n_iter_ = current_init_n_iter
                self.lower_bound_ = current_init_lower_bound
                self.parameters_ = parameters
                if self.save_history:
                    self.history = init_history
        return self

    def update(self, x, alpha):
        self.parameters_ = self.base_distribution.update(x, self.parameters_, alpha)

    def predict(self, x):
        return np.argmax(self.base_distribution.logpdf(x, self.parameters_), axis=1)

    def predict_proba(self, x):
        return np.exp(self._e_step(x, self.parameters_)[1])

    def predict_distance(self, x):
        return self.base_distribution.distance(x, self.parameters_)

    def predict_likelihood(self, x):
        return scipy.special.logsumexp(self.base_distribution.logpdf(x, self.parameters_), axis=1)


class BGSegmenter:
    def __init__(self, base_clusterizer, *clusterizer_args, **clusterizer_kwargs):
        self.base_clusterizer = base_clusterizer
        self.clusterizer_args = clusterizer_args
        self.clusterizer_kwargs = clusterizer_kwargs

        self.models = []

    def fit(self, x):
        """
        :param x: Sequence of images
            Each image presented as squared array of pixels
            Each pixel presented as array of features
        :return: self
        """
        if len(x.shape) == 3:
            x = x[:, :, :, np.newaxis]
        n_o, x_w, x_h, n_f = x.shape
        x = np.ascontiguousarray(x.transpose([1, 2, 0, 3]).reshape([x_w * x_h, n_o, n_f]))
        for idx, pixel in tqdm(enumerate(x), total=x.shape[0]):
            self.models.append(
                self.base_clusterizer(
                    *self.clusterizer_args, **self.clusterizer_kwargs
                ).fit(pixel)
            )
        return self

    def predict_update(self, x, way, threshold, alpha):
        """
            Adaptive predictions
        :param x:
        :param way:
        :param threshold:
        :param alpha:
        :return:
        """
        if len(x.shape) == 3:
            x = x[:, :, :, np.newaxis]
        result = np.empty(x.shape[:-1])
        for idx, image in tqdm(enumerate(x), total=x.shape[0]):
            distance = None
            if way == 'distance':
                distance = self.predict_distance(x[idx:idx + 1], disable=True)
            if way == 'likelihood':
                distance = self.predict_likelihood(x[idx:idx + 1], disable=True)
            result[idx] = distance
            update_mask = distance < threshold
            for pixel, is_update, model in zip(
                    image.reshape(-1, image.shape[-1]), update_mask.reshape(-1), self.models
            ):
                model.update(pixel, alpha)
        return result

    def predict_distance(self, x, disable=False):
        if len(x.shape) == 3:
            x = x[:, :, :, np.newaxis]
        n_o, x_w, x_h, n_f = x.shape
        x = np.ascontiguousarray(x.transpose([1, 2, 0, 3]).reshape([x_w * x_h, n_o, n_f]))
        result = []
        for pixel, model in tqdm(zip(x, self.models), total=x.shape[0], disable=disable):
            result.append(model.predict_distance(pixel))
        return np.ascontiguousarray(np.array(result).reshape([x_w, x_h, n_o]).transpose([2, 0, 1]))

    def predict_likelihood(self, x, disable=False):
        if len(x.shape) == 3:
            x = x[:, :, :, np.newaxis]
        n_o, x_w, x_h, n_f = x.shape
        x = np.ascontiguousarray(x.transpose([1, 2, 0, 3]).reshape([x_w * x_h, n_o, n_f]))
        result = []
        for pixel, model in tqdm(zip(x, self.models), total=x.shape[0], disable=disable):
            result.append(model.predict_likelihood(pixel))
        return np.ascontiguousarray(np.array(result).reshape([x_w, x_h, n_o]).transpose([2, 0, 1]))

    def get_background(self, shape):
        result = []
        for model in self.models:
            result.append(model.base_distribution.sample(model.parameters_))
        return np.array(result).reshape(shape)
