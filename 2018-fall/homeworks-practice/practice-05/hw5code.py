import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    x = feature_vector
    y = target_vector

    idx_sorted = np.argsort(x, axis=0)

    x_sorted = np.sort(x, axis=0)
    y_sorted = y[idx_sorted]

    n = feature_vector.shape[0]
    n_0_i = np.cumsum(y_sorted == 0)
    n_0 = n_0_i[-1]
    n_0_i = n_0_i[:-1]

    i = np.array(range(1, n)).astype(np.float64)

    gini = 2 / n / (n - i) * (n_0 * (i - (n - n_0)) - n_0_i * (2 * n_0 - n * n_0_i / i))
    gini = gini[(~(x_sorted == np.roll(x_sorted, shift=-1, axis=0)))[:-1]]

    n_best_obj = np.argmax(gini)
    x_sorted_unique = np.unique(x_sorted)
    thresholds = (x_sorted_unique[:-1] + x_sorted_unique[1:]) / 2

    return thresholds, gini, thresholds[n_best_obj],  np.max(gini)


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")
        super().__init__()
		
        self._tree = {}
        self.feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_x, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_x.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_x[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_x[:, feature])
                clicks = Counter(sub_x[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_x[:, feature])))
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_x[split], sub_y[split], node["left_child"])
        self._fit_node(sub_x[~split], sub_y[~split], node["right_child"])

    def _predict_node(self, x, node):
        while node["type"] != "terminal":
            feature_split = node["feature_split"]
            next_node = "left_child"
            if self.feature_types[feature_split] == "real":
                if x[feature_split] >= node["threshold"]:
                    next_node = "right_child"
            elif self.feature_types[feature_split] == "categorical":
                if x[feature_split] not in node["categories_split"]:
                    next_node = "right_child"
            else:
                raise ValueError
            node = node[next_node]
        return node["class"]

    def fit(self, x, y):
        self._fit_node(x, y, self._tree)

    def predict(self, x):
        predicted = []
        for x in x:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
