# Codveda Technology - Machine Learning Internship

My completed tasks for the Codveda Machine Learning internship, built with Python, pandas, scikit-learn and TensorFlow/Keras in Google Colab. Every notebook is saved with its outputs and charts.

**Video walkthrough:** [ADD YOUR LINKEDIN VIDEO LINK]
**LinkedIn post:** [ADD YOUR LINKEDIN POST LINK]

## Project structure

```
codveda-ml-internship/
├── level1/   (Basic)
│   ├── Task1_Task2_Preprocessing_LinearRegression.ipynb
│   └── Task3_KNN_Classifier.ipynb
├── level2/   (Intermediate)
│   └── Level2_LogReg_DecisionTree_KMeans.ipynb
├── level3/   (Advanced)
│   └── Level3_RandomForest_SVM_NeuralNetwork.ipynb
├── requirements.txt
└── README.md
```

## Tasks, datasets and results

| Level | Task | Dataset | Result |
|---|---|---|---|
| 1 | Data Preprocessing | Kaggle House Prices (1,460 rows, 80 columns) | Columns with more than 40% missing values dropped, the rest imputed (median / most frequent), one-hot encoding + standardization (73 -> 264 features), 80/20 split |
| 1 | Linear Regression | Kaggle House Prices | R² = 0.885, RMSE = $29,680 |
| 1 | KNN Classifier | Breast Cancer Wisconsin (scikit-learn) | Accuracy = 0.982 at K = 3; K from 7 to 15 gives a stable 0.974 |
| 2 | Logistic Regression | Breast Cancer Wisconsin | Accuracy = 0.982, precision = 0.986, recall = 0.986, AUC = 0.995 |
| 2 | Decision Tree | Iris | Pruned tree (depth 3, 5 leaves): accuracy = F1 = 0.978, vs 0.933 for the unpruned tree (train accuracy 1.0, overfitting) |
| 2 | K-Means Clustering | Wine | k = 3 clusters, silhouette = 0.285 |
| 3 | Random Forest | Breast Cancer Wisconsin | Best params: 200 trees, no max depth; CV F1 = 0.968 (+/- 0.015); test accuracy = 0.96 |
| 3 | SVM | Breast Cancer Wisconsin | Linear: accuracy 0.974, AUC 0.996. RBF: accuracy 0.982, AUC 0.995 |
| 3 | Neural Network (Keras) | MNIST | Test accuracy = 0.9771 (10 epochs) |

## How to run

1. Open [Google Colab](https://colab.research.google.com) and upload a notebook (File > Upload notebook).
2. Run all cells (Runtime > Run all).
3. The Level 1 Task 1/2 notebook asks you to upload `train.csv` from the Kaggle "House Prices - Advanced Regression Techniques" competition. All other notebooks use datasets built into scikit-learn or Keras.

To run locally instead: `pip install -r requirements.txt` and open the notebooks in Jupyter.

## What I learned

[Write 3-5 sentences: e.g. why the train/test split must come before scaling (data leakage), how pruning reduces overfitting, how the RBF kernel and the neural network capture non-linear patterns.]

## Contact

[Your name] - [LinkedIn profile link]
Internship by [@Codveda](https://www.linkedin.com/company/codveda)
