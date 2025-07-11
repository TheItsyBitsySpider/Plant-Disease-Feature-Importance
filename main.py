import pandas as pd
import numpy
import sklearn
import xgboost
import imblearn
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss

all_data = pd.read_csv("./plant_disease_dataset.csv")
disease_labels = all_data.pop("disease_present")


boost_tree = XGBClassifier(verbosity=2)

x_train, x_test, y_train, y_test = train_test_split(all_data, disease_labels, test_size=.20, random_state=42, shuffle=True, stratify=disease_labels)
sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

sampler = SMOTE()

# Plot class distribution to ensure that train/test/val split has equal proportions for proper testing & training
plt.bar(x=y_train, height=len(y_train), color="red")
plt.bar(x=y_test, height=len(y_test), color="green")
plt.ylabel("Amount")
plt.title("Distribution of Data")

plt.savefig("AmountOfClasses.png")
plt.clf()

print(f"Dataset containing {len(all_data)} values")
print(f"Splits containing {len(y_train)} train vals and {len(y_test)} test vals")

model = make_pipeline(sampler, boost_tree)

search_for_best_params = GridSearchCV(model, param_grid= {
    "smote__sampling_strategy": ["majority", "not minority", "not majority", "all"],
    "smote__k_neighbors": [1, 3, 5],
    "xgbclassifier__learning_rate": [0.025, 0.05, 1],
    "xgbclassifier__lambda": [1, 1.25, 1.5, 1.75],
    "xgbclassifier__max_delta_step": [0, 1, 2],
    "xgbclassifier__gamma": [0, 1, 2, 4, 5, 6],
}, n_jobs=24, verbose=3)

search_for_best_params.fit(x_train, y_train)

print(f"Best Features: {search_for_best_params.best_params_}")

best_estimator = search_for_best_params.best_estimator_

y_pred = best_estimator.predict(x_test)

print(classification_report(y_test, y_pred))
print(f"ROC AUC score: {roc_auc_score(y_score=y_pred, y_true=y_test)}")


perm_importance = permutation_importance(best_estimator, x_test, y_test)

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(all_data.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.savefig("FeatureImportance.png")