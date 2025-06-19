import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict


class DecisionTreeClassifier:
    def fit(self, X, y):
        data = pd.DataFrame(X.copy())
        data['label'] = y
        self.tree = self._build_tree(data)

    def predict(self, X):
        return X.apply(self._predict_row, axis=1)

    def _entropy(self, y):
        counts = Counter(y)
        total = len(y)
        return -sum((count / total) * np.log2(count / total + 1e-9) for count in counts.values())

    def _best_split(self, data):
        base_entropy = self._entropy(data['label'])
        best_info_gain = -1
        best_feature = None
        best_threshold = None

        for feature in data.columns[:-1]:
            if data[feature].dtype.kind in "bifc":  # 数值型特征
                sorted_values = data[feature].sort_values().unique()
                thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
                for threshold in thresholds:
                    below = data[data[feature] <= threshold]
                    above = data[data[feature] > threshold]
                    if below.empty or above.empty:
                        continue
                    new_entropy = (len(below) / len(data)) * self._entropy(below['label']) + \
                                  (len(above) / len(data)) * self._entropy(above['label'])
                    info_gain = base_entropy - new_entropy
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_feature = feature
                        best_threshold = threshold
            else:
                values = data[feature].unique()
                new_entropy = sum((len(subset := data[data[feature] == val]) / len(data)) * self._entropy(subset['label'])
                                  for val in values)
                info_gain = base_entropy - new_entropy
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = None

        return best_feature, best_threshold

    def _build_tree(self, data):
        if len(set(data['label'])) == 1:
            return data['label'].iloc[0]
        if len(data.columns) == 1:
            return data['label'].mode()[0]

        best_feature, threshold = self._best_split(data)
        if best_feature is None:
            return data['label'].mode()[0]

        tree = {}
        if threshold is not None:
            tree = {f"{best_feature}<={threshold:.4f}": {}}
            below = data[data[best_feature] <= threshold].drop(columns=[best_feature])
            above = data[data[best_feature] > threshold].drop(columns=[best_feature])
            tree[f"{best_feature}<={threshold:.4f}"]['yes'] = self._build_tree(below)
            tree[f"{best_feature}<={threshold:.4f}"]['no'] = self._build_tree(above)
        else:
            tree = {best_feature: {}}
            for value in data[best_feature].unique():
                subtree = self._build_tree(data[data[best_feature] == value].drop(columns=[best_feature]))
                tree[best_feature][value] = subtree
        return tree

    def _predict_row(self, row):
        tree = self.tree
        while isinstance(tree, dict):
            node = next(iter(tree))
            if '<=' in node:
                feature, thresh = node.split('<=')
                thresh = float(thresh)
                tree = tree[node]['yes'] if row[feature] <= thresh else tree[node]['no']
            else:
                value = row[node]
                tree = tree[node].get(value, list(tree[node].values())[0])
        return tree

    def get_feature_importance(self):
        importance = defaultdict(int)

        def traverse(node):
            if not isinstance(node, dict):  # 叶节点
                return
            feature = next(iter(node))
            importance[feature] += 1
            branches = node[feature]
            for key in branches:
                traverse(branches[key])

        traverse(self.tree)
        total = sum(importance.values())
        return {feature: count / total for feature, count in importance.items()}