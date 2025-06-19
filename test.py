import pandas as pd
import numpy as np
import os
import pickle
from train import train_models, load_data

def load_or_train_models(dataset):
    weights_dir = os.path.join("models", "weights")
    knn_path = os.path.join(weights_dir, f"{dataset}_knn.pkl")
    tree_path = os.path.join(weights_dir, f"{dataset}_tree.pkl")
    nb_path = os.path.join(weights_dir, f"{dataset}_nb.pkl")
    cnn_path = os.path.join(weights_dir, f"{dataset}_cnn.pkl")  # 加入CNN路径

    if all(os.path.exists(p) for p in [knn_path, tree_path, nb_path, cnn_path]):
        with open(knn_path, "rb") as f:
            knn = pickle.load(f)
        with open(tree_path, "rb") as f:
            tree = pickle.load(f)
        with open(nb_path, "rb") as f:
            nb = pickle.load(f)
        with open(cnn_path, "rb") as f:
            cnn = pickle.load(f)
    else:
        knn, tree, nb, cnn = train_models(dataset)

    return knn, tree, nb, cnn

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    correct = sum(p == t for p, t in zip(y_pred, y_test))
    return correct / len(y_test)


def main():
    datasets = ['iris', 'wine', 'breast_cancer', 'digits']
    for dataset in datasets:
        print(f"\nEvaluating on {dataset} dataset:")
        X_test, y_test = load_data(f"data/{dataset}_test.csv")
        knn, tree, nb, cnn = load_or_train_models(dataset)

        print("KNN Accuracy:", evaluate_model(knn, X_test, y_test))
        print("Decision Tree Accuracy:", evaluate_model(tree, X_test, y_test))
        print("Naive Bayes Accuracy:", evaluate_model(nb, X_test.values, y_test.values))
        print("CNN Accuracy:", evaluate_model(cnn, X_test, y_test))

if __name__ == "__main__":
    main()
