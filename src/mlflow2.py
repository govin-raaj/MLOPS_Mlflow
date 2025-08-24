import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='govin-raaj', repo_name='MLOPS_Mlflow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/govin-raaj/MLOPS_Mlflow.mlflow")

wine = load_wine()
x=wine.data
y=wine.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

max_depth = 8
n_estimators = 100

mlflow.set_experiment("Experiment_2")
with mlflow.start_run():
    clf=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    print(f"Model accuracy: {accuracy}")