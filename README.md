# Plant-Disease-Feature-Importance

This small project focuses on analyzing the following plant disease dataset from Kaggle & figuring out which features are most important from it

https://www.kaggle.com/datasets/turakut/plant-disease-classification

The model is a gradient boosted decision tree using the XGBoost library that tries to predict whether disease is present given the dataset's features. Below are the metrics used to judge the model

              precision    recall  f1-score   support

           0       0.89      0.94      0.92      1518
           1       0.78      0.64      0.70       482

    accuracy                           0.87      2000

ROC AUC score: 0.7915047097349098

Overall, it performed adequately, especially given the imbalanced nature of the dataset, with 25% being diseased plants & 75% being non-diseased

Below is a chart of feature importance, utilizing permutation_importance from Scikit-Learn

<img width="800" height="800" alt="FeatureImportance" src="https://github.com/user-attachments/assets/81647002-6710-4967-b0ff-bb9453d745f3" />

According to the findings, soil PH is the biggest factor in determining whether a plant will end up being diseased or not, and temperature being the least important factor
