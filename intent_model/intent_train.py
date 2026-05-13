"""
Intent Classifier Training Pipeline
=====================================
Trains a lightweight intent classifier used by the Tajstack AI
to detect high-risk customer intents before generation.

Architecture:
    MiniLM (sentence-transformers) → embeddings → Logistic Regression → intent label

Why this approach instead of fine-tuning a full transformer:
    - MiniLM embeddings are high quality and generalize well
    - Logistic Regression on top is fast to train, fast to infer, and interpretable
    - Full fine-tuning would require far more labeled data and GPU resources
    - This approach achieves strong accuracy on 5-class classification
      with a fraction of the complexity

The classifier is used in production to flag intents like refund_request
or complaint before retrieval — allowing the system to escalate or apply
stricter policies before any generation happens.

Usage:
    python intent_train.py

Output:
    intent_classifier.joblib   ← trained Logistic Regression classifier
    intent_model/              ← saved MiniLM embedder
"""

import numpy as np
import joblib
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# -----------------------
# LABEL MAPPING
# -----------------------
# 5-class intent classes covering the most common high-risk
# and operational customer message types.
label_list = [
    "refund_request",   # triggers hard escalation for high-risk clients
    "order_status",     # informational, low risk
    "technical_issue",  # may escalate depending on severity
    "complaint",        # triggers hard escalation for high-risk clients
    "general_query",    # default fallback intent
]

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}


# -----------------------
# LOAD DATASET
# -----------------------
dataset = load_dataset("csv", data_files="intent_dataset.csv")

# 80/20 train-test split
dataset = dataset["train"].train_test_split(test_size=0.2)


def encode_labels(batch):
    """Converts string labels to integer IDs for sklearn."""
    batch["label"] = label2id[batch["label"]]
    return batch


dataset = dataset.map(encode_labels)


# -----------------------
# EMBEDDINGS
# -----------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding training set...")
X_train = embedder.encode(dataset["train"]["text"], convert_to_numpy=True)
X_test = embedder.encode(dataset["test"]["text"], convert_to_numpy=True)

y_train = np.array(dataset["train"]["label"])
y_test = np.array(dataset["test"]["label"])

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")


# -----------------------
# TRAIN CLASSIFIER
# -----------------------
# Logistic Regression with high max_iter to ensure convergence
# on the 384-dimensional embedding space.
# No regularization tuning needed at this dataset scale —
# default C=1.0 generalizes. This is because the dataset is already balanced.
print("Training Logistic Regression classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


# -----------------------
# EVALUATION
# -----------------------
y_pred = clf.predict(X_test)

print("\n--- Classification Report ---\n")
print(classification_report(y_test, y_pred, target_names=label_list))

print("--- Confusion Matrix ---\n")
print(confusion_matrix(y_test, y_pred))

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")


# -----------------------
# SAVE
# -----------------------
# Classifier saved as joblib — loaded at app.py startup via lazy-loading.
# Embedder saved as a directory — loaded by SentenceTransformer at runtime inside app.py.
joblib.dump(clf, "intent_classifier.joblib")
embedder.save("intent_model_embedder")

print("\nSaved: intent_classifier.joblib")
print("Saved: intent_model/")
