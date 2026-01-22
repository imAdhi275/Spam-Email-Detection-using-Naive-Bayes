import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)

def evaluate_model(model, model_type, X_test_vec, y_test, exp_name, output_dir):
    if model_type == "gaussian":
        y_pred = model.predict(X_test_vec.toarray())
        y_proba = model.predict_proba(X_test_vec.toarray())[:, 1]
    else:
        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec)[:, 1]

    print("\n==============================")
    print(f"RESULTS: {exp_name}")
    print("==============================")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Ham","Spam"]).plot()
    plt.title(exp_name + " - Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"{exp_name}_CM.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title(exp_name + " - ROC Curve")
    plt.savefig(os.path.join(output_dir, f"{exp_name}_ROC.png"))
    plt.close()

    return {
        "Experiment": exp_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    }