import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from ImageTransformer import ImageTransformer
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

## BUILD OUR PIPELINE

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

verbose = 2

# make pipeline out of our classifier and preprocessor
pipe_itf = Pipeline(steps = [
                        ('itf', ImageTransformer(32)),
                        ('classifier', RandomForestClassifier(max_depth = 3, random_state=0))])


itf_grid = {
    'itf__size':[30,32],
    'itf__interpolation':[cv2.INTER_LINEAR]
}

classifier_grid = {
    'classifier__max_depth':[4,5],
    'classifier__min_samples_split':[2, 0.005],
    'classifier__n_estimators': [100,200,300]

}

pca_grid = {
    'pca__n_components':[1.0,0.95,0.9]
}

grid_search_itf = GridSearchCV(
    estimator = pipe_itf,
    param_grid = itf_grid | classifier_grid,
    cv = 5,
    scoring = 'f1',
    verbose = verbose,
    n_jobs = -1
)

## ITS SHOW TIME BABY
# format our data
df = pd.read_csv('C:/Users/user/Documents/Academic/Fall 2025/ML I/final_project/dataset_index.csv')

df = df[df['label'] == 'Moose']

target = 'day'
X = df.drop(columns=[target])
y = df[target]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# run grid search
print('\n\n' + '='*10 + 'PIPELINE 1: ITF' + '='*10)
grid_search_itf.fit(X_train, y_train)
print('\n\n' + '='*10 + "GRID SEARCH RESULTS" + '='*10)
print(f"Best params: {grid_search_itf.best_params_}")
print(f"Best cross-validation score (accuracy): {grid_search_itf.best_score_:.4f}")


# run inference on test data
y_pred = grid_search_itf.best_estimator_.predict(X_test)

# performance report
print(f"\n\nMODEL PERFORMANCE:")
print("\n" + classification_report(y_test, y_pred))

# export to pkl
import joblib
export_path = 'C:/Users/user/Documents/Academic/Fall 2025/ML I/final_project/daynight_classifier.pkl'


print("\n\nExporting model...")
joblib.dump(grid_search_itf.best_estimator_, export_path)
print("Export complete!")


