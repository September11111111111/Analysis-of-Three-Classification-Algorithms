import pandas as pd
import numpy as np
import os
import pickle
from models.knn import KNNClassifier
from models.decision_tree import DecisionTreeClassifier
from models.naive_bayes import NaiveBayesClassifier
from models.cnn import CNNClassifier

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y

def train_models(dataset_name):
    base_path = os.path.join("data")
    X_train, y_train = load_data(os.path.join(base_path, f"{dataset_name}_train.csv"))

    # 初始化模型
    knn = KNNClassifier(k=3)
    tree = DecisionTreeClassifier()
    nb = NaiveBayesClassifier()
    cnn = CNNClassifier()

    # 训练模型
    knn.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    nb.fit(X_train.values, y_train.values)
    cnn.fit(X_train, y_train)

    # 保存模型权重
    weights_dir = os.path.join("models", "weights")
    os.makedirs(weights_dir, exist_ok=True)

    with open(os.path.join(weights_dir, f"{dataset_name}_knn.pkl"), "wb") as f:
        pickle.dump(knn, f)
    with open(os.path.join(weights_dir, f"{dataset_name}_tree.pkl"), "wb") as f:
        pickle.dump(tree, f)
    with open(os.path.join(weights_dir, f"{dataset_name}_nb.pkl"), "wb") as f:
        pickle.dump(nb, f)
    with open(os.path.join(weights_dir, f"{dataset_name}_cnn.pkl"), "wb") as f:   # **新增：保存CNN模型**
        pickle.dump(cnn, f)

    return knn, tree, nb, cnn
