# Codveda Technology - Machine Learning Internship

My completed tasks for the Codveda Machine Learning internship, built with Python, pandas, scikit-learn and TensorFlow/Keras in Google Colab. Every notebook is saved with its outputs and charts.


**LinkedIn post:** (https://lnkd.in/p/e29NEuGu)

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

## What I learned

- **Data preparation comes first.** I learned to split the data into train and test sets *before* imputing and scaling, so that no information from the test set leaks into training. Handling missing values, one-hot encoding and standardization turned the raw House Prices data (73 columns) into 264 clean numeric features.
- **A good score is not enough, I also have to interpret it.** My linear regression reached R² = 0.885, but the largest coefficients came from very rare categories, so I learned to be careful when reading one-hot coefficients. For logistic regression, odds ratios gave me a way to explain which features push a prediction up or down.
- **Overfitting is real and measurable.** An unpruned decision tree scored 100% on training data but only 93.3% on test data. After pruning, the tree was simpler and reached 97.8% on test. For KNN, choosing K showed me the trade-off between a model that is too flexible and one that is too smooth.
- **Accuracy alone can hide mistakes.** On the tumor classification task I looked at precision, recall, the confusion matrix and the ROC curve, because missing a malignant tumor matters more than a small drop in accuracy.
- **Tuning and validation make results trustworthy.** With GridSearchCV and cross-validation, my Random Forest reached an F1 score of 0.968. I also compared linear and RBF kernels for the SVM (97.4% vs 98.2% accuracy) and saw that the extra complexity only helps a little on this dataset.
- **Neural networks learn from raw pixels.** A simple Keras network reached 97.71% test accuracy on MNIST, and the training and validation loss curves helped me check for overfitting.

Next, I want to practice feature engineering, handle imbalanced data, and try convolutional neural networks.

## Contact

Asmae Bihi - https://www.linkedin.com/in/asmae-bihi/.
Internship by [@Codveda](https://www.linkedin.com/company/codveda)
