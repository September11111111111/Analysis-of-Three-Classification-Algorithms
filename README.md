# 分类算法对比实验

## 项目结构
- `data/`: 存放训练与测试 CSV 文件
- `models/`: 自实现的 KNN、决策树、朴素贝叶斯分类器
- `train.py`: 模型训练脚本
- `test.py`: 模型评估脚本

## 环境安装
```bash
pip install -r requirements.txt
```

## 运行方式
```bash
python test.py
```

## 支持数据集
- iris
- wine
- breast_cancer
- digits

## 查看权重
```bash
python Read_pkl.py
```
