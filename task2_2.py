import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
(X_iris, y_iris) = load_iris(return_X_y=True)
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=0)
clf = LogisticRegression(C=1e1, solver="lbfgs")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)

print("Logistic regression classifier \n ", clf)
print(
    f"Accuracy of LR classifier on training set: {clf.score(X_train_scaled, y_train):.2f}"
)
print(f"Accuracy of LR classifier on test set: {clf.score(X_test_scaled, y_test):.2f}")
matrix = confusion_matrix(y_test, predictions)  # ,labels)
print(matrix)
print(classification_report(y_test, predictions))
print("training results")
predictions = clf.predict(X_train_scaled)
print(classification_report(y_train, predictions))
print("***Explaination***")
##### Using lime

feature_list = iris.feature_names
model_pred = clf.predict(X_test_scaled)
# Create a lime explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    mode="classification",
    training_labels=y_train,
    feature_names=feature_list,
)
labels = {0: "setosa", 1: "versicolor", 2: "virginica"}
print(f"Correct class: {model_pred[13]} - {labels[model_pred[13]]}")
print(
    clf.predict_proba(
        [
            X_test_scaled[13],
        ]
    )
)
exp = explainer.explain_instance(
    X_test_scaled[13], predict_fn=clf.predict_proba, num_features=4, top_labels=2
)

fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.show()
exp.save_to_file("lime2.html")
