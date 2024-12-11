import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Paths to results
NN_RESULTS_PATH = "results/nn_predictions.csv"
TRANSFORMER_RESULTS_PATH = "results/transformer_predictions.csv"

# Load prediction results
nn_results = pd.read_csv(NN_RESULTS_PATH)
transformer_results = pd.read_csv(TRANSFORMER_RESULTS_PATH)

# Extract true and predicted labels
nn_y_true = nn_results["true_label"]
nn_y_pred = nn_results["predicted_label"]

transformer_y_true = transformer_results["true_label"]
transformer_y_pred = transformer_results["predicted_label"]

# Define a function to calculate metrics
def calculate_metrics(y_true, y_pred):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1 Score": f1_score(y_true, y_pred, average="weighted")
    }
    return metrics

# Calculate metrics for both models
nn_metrics = calculate_metrics(nn_y_true, nn_y_pred)
transformer_metrics = calculate_metrics(transformer_y_true, transformer_y_pred)

# Print metrics
print("Neural Network Metrics:")
print(nn_metrics)
print("\nClassification Report:")
print(classification_report(nn_y_true, nn_y_pred))

print("\nTransformer Metrics:")
print(transformer_metrics)
print("\nClassification Report:")
print(classification_report(transformer_y_true, transformer_y_pred))

# Compare metrics
comparison_df = pd.DataFrame({
    "Metric": list(nn_metrics.keys()),
    "Neural Network": list(nn_metrics.values()),
    "Transformer": list(transformer_metrics.values())
})
print("\nComparison of Metrics:")
print(comparison_df)

# Save the comparison to a CSV
comparison_df.to_csv("results/model_comparison.csv", index=False)
print("\nComparison saved to results/model_comparison.csv")
