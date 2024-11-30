import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = MLPClassifier(
    hidden_layer_sizes=(48, 8),
    activation="relu",
    max_iter=1000,
    solver="lbfgs",
    random_state=0,
)


clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)

print("MLP classifier \n ", clf)
print(
    f"Accuracy of MLP classifier on training set: {clf.score(X_train_scaled, y_train):.2f}"
)
print(f"Accuracy of MLP classifier on test set: {clf.score(X_test_scaled, y_test):.2f}")

matrix = confusion_matrix(y_test, predictions)  # ,labels)
print(matrix)
print(classification_report(y_test, predictions))
print("training results")
predictions = clf.predict(X_train_scaled)
print(classification_report(y_train, predictions))
print("***Explaination***")
##### Using lime

feature_list = cancer.feature_names


model_pred = clf.predict(X_test_scaled)
# Create a lime explainer object

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    mode="classification",
    training_labels=y_train,
    feature_names=feature_list,
)
labels = {0: "malignant", 1: "benign"}
print(f"Correct class: {model_pred[3]} - {labels[model_pred[3]]}")
print(
    clf.predict_proba(
        [
            X_test_scaled[3],
        ]
    )
)
exp = explainer.explain_instance(
    X_test_scaled[3], predict_fn=clf.predict_proba, num_features=10
)

fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.show()
# exp.save_to_file("lime.html")
