import pickle

def print_knn_info(knn):
    print("KNN 模型对象:", knn)
    # 假设你的 KNN 有保存训练数据X_train, y_train
    print("X_train 形状:", knn.X_train.shape)
    print("y_train 形状:", knn.y_train.shape)

def print_tree_info(tree):
    print("决策树模型对象:", tree)
    # 假设决策树自定义了 get_feature_importance 方法
    if hasattr(tree, "get_feature_importance"):
        print("特征重要性:", tree.get_feature_importance())
    else:
        print("该决策树模型没有 get_feature_importance 方法")

def print_nb_info(nb):
    print("朴素贝叶斯模型对象:", nb)
    print("类别:", nb.classes)
    print("\n均值 (mean):")
    for c in nb.classes:
        print(f"类别 {c}: {nb.mean[c]}")
    print("\n方差 (var):")
    for c in nb.classes:
        print(f"类别 {c}: {nb.var[c]}")
    print("\n先验概率 (priors):")
    for c in nb.classes:
        print(f"类别 {c}: {nb.priors[c]}")

def print_model_info(dataset_name):
    print(f"\n====== {dataset_name} 数据集模型权重信息 ======")

    knn_path = f"models/weights/{dataset_name}_knn.pkl"
    tree_path = f"models/weights/{dataset_name}_tree.pkl"
    nb_path = f"models/weights/{dataset_name}_nb.pkl"

    # 载入knn
    with open(knn_path, "rb") as f:
        knn = pickle.load(f)
    print_knn_info(knn)

    # 载入tree
    with open(tree_path, "rb") as f:
        tree = pickle.load(f)
    print_tree_info(tree)

    # 载入nb
    with open(nb_path, "rb") as f:
        nb = pickle.load(f)
    print_nb_info(nb)

if __name__ == "__main__":
    for dataset in ["breast_cancer", "iris", "wine"]:
        print_model_info(dataset)
