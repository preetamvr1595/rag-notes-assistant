"""Generate confusion matrix for out-of-scope refusal gate."""

def calculate_confusion_matrix(tp=28, fp=1, tn=11, fn=3):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print("=== Refusal Gate Confusion Matrix ===")
    print(f"True In-Scope Answered (TP):  {tp}")
    print(f"False Out-of-Scope Ans (FP): {fp}")
    print(f"True Out-of-Scope Refused (TN): {tn}")
    print(f"False In-Scope Refused (FN): {fn}")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")

if __name__ == "__main__":
    calculate_confusion_matrix()
