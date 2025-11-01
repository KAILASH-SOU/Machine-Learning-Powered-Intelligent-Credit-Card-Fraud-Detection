from utils import load_model, load_data, preprocess_data, split_data
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate():
    model = load_model("fraud_model.pkl")
    df = load_data("creditcard.csv")
    X, y = preprocess_data(df)
    _, X_test, _, y_test = split_data(X, y)

    y_pred = model.predict(X_test)

    print("\n✅ Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    evaluate()
