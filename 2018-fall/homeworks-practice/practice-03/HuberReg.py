import numpy as np

from sklearn.base import BaseEstimator


class HuberReg(BaseEstimator):
    def __init__(self, delta=1.0, gd_type='stochastic', 
                 tolerance=1e-4, max_iter=1000, w0=None, alpha=1e-3, eta=1e-2):
        """
        gd_type: 'full' or 'stochastic'
        tolerance: for stopping gradient descent
        max_iter: maximum number of steps in gradient descent
        w0: np.array of shape (d) - init weights
        eta: learning rate
        alpha: momentum coefficient
        """
        self.delta = delta
        self.gd_type = gd_type
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w0 = w0
        self.alpha = alpha
        self.w = None
        self.eta = eta
        self.loss_history = None # list of loss function values at each training iteration
        self.score_history = None
    
    def fit(self, X, y, verbose=False):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: self
        """
        self.loss_history = []
        self.score_history = []
        
        if self.w0 is None:
            self.w0 = np.zeros([X.shape[1]], dtype=np.float64)
        w = self.w = self.w0

        n_iter = 0
        delta = np.zeros([X.shape[1]])
        while n_iter < self.max_iter and (np.linalg.norm(w - self.w) > self.tolerance or n_iter == 0):
            n_iter += 1
            self.w = w
            if self.gd_type == 'stochastic':
                idxs = np.random.randint(0, X.shape[0], [1])
            else:
                idxs = range(0, X.shape[0])
            self.loss_history.append(self.calc_loss(X, y))
            self.score_history.append(self.score(X, y))
            delta = -self.eta * self.calc_gradient(X[idxs], y[idxs]) + self.alpha * delta
            w = self.w + delta
            if verbose:
                print(self.loss_history[-1])
        self.w = w
        self.loss_history.append(self.calc_loss(X, y))
        self.score_history.append(self.score(X, y))
        
        return self
    
    def predict(self, X):
        if self.w is None:
            raise Exception('Not trained yet')
        return np.matmul(X, self.w)
        
    def calc_gradient(self, X, y):
        """
        X: np.array of shape (l, d) (l can be equal to 1 if stochastic)
        y: np.array of shape (l)
        ---
        output: np.array of shape (d)
        """
        delta = self.predict(X) - y
        delta[np.abs(delta) > self.delta] = self.delta * np.sign(delta[np.abs(delta) > self.delta])
        return np.matmul(X.T, delta / float(delta.shape[0]))

    def calc_loss(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: float 
        """ 
        delta = (self.predict(X) - y)
        result = 0.5 * delta ** 2
        result[np.abs(delta) > self.delta] = self.delta * np.abs(delta[np.abs(delta) > self.delta]) - 0.5 * self.delta ** 2
        return np.mean(result)
    
    def score(self, X, y):
        from sklearn.metrics import r2_score
        predicts = self.predict(X)
        return r2_score(y, predicts)