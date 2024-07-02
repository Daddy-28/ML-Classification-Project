import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('finaltrain6.csv')

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Performance']), 
                                                    data['Performance'], test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier()

# Define a broader search space for hyperparameters (using param_distributions)
param_distributions = {
    'learning_rate': np.arange(0.01, 0.3, 0.01),  # Explore finer learning rates
    'max_depth': range(3, 8),
    'subsample': np.arange(0.6, 1, 0.05),
    'colsample_bytree': np.arange(0.6, 1, 0.05)
}

# Use RandomizedSearchCV for efficient hyperparameter tuning
rand_search = RandomizedSearchCV(estimator=xgb_classifier, param_distributions=param_distributions, cv=5, n_jobs=-1)
rand_search.fit(X_train, y_train)

# Get the best parameters and estimator
best_params = rand_search.best_params_
best_estimator = rand_search.best_estimator_

print("Best Parameters:", best_params)

# Predict using the best estimator
y_pred = best_estimator.predict(X_test_imputed)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy after Hyperparameter Tuning: {accuracy}")

# Initialize SVM classifier
svm_classifier = SVC()

# Fit SVM classifier
svm_classifier.fit(X_train_imputed, y_train)

# Initialize LGBM classifier
lgbm_classifier = LGBMClassifier()

# Fit LGBM classifier
lgbm_classifier.fit(X_train_imputed, y_train)

# Initialize voting classifier
voting_classifier = VotingClassifier(estimators=[
    ('xgb', best_estimator),
    ('svm', svm_classifier),
    ('lgbm', lgbm_classifier)
], voting='hard')  # 'hard' voting for simple majority

# Fit voting classifier
voting_classifier.fit(X_train_imputed, y_train)

# Predict using the voting classifier
y_pred_ensemble = voting_classifier.predict(X_test_imputed)

# Calculate accuracy
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Accuracy: {accuracy_ensemble}")
f1_ens = f1_score(y_test, y_pred_ensemble)
print(f"SVM F1 Score: {f1_ens}")

# Now make predictions on the new data using the ensemble
dfg = pd.read_csv('newtest2.csv')
features = ['WORK', 'GRAD', 'EXP', 'PrevWork', 'INCENT', 'Ref', 'products', 'ticket', 'member', 'depend', 'Department', 'RC', 'lang','PREVIND','NOIND']
ll_test = dfg[features]
ll_test_imputed = imputer.transform(ll_test)
Y_pred_ensemble = voting_classifier.predict(ll_test_imputed)

# Get feature importances from XGBoost
xgb_feature_importances = best_estimator.feature_importances_

# Get feature importances from LightGBM
lgbm_feature_importances = lgbm_classifier.feature_importances_

# Combine feature importances from both models
feature_importances = { 'LightGBM': xgb_feature_importances}

# Sort feature importances in ascending order
sorted_feature_importances = sorted(zip(features, xgb_feature_importances), key=lambda x: x[1])

# Extract sorted features and importances
sorted_features, sorted_importances = zip(*sorted_feature_importances)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance of LightGBM (Ascending Order)')
plt.show()

# Print feature importances in ascending order
print("Feature Importance in Ascending Order:")
for feature, importance in sorted_feature_importances:
    print(f"{feature}: {importance}")
