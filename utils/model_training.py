# ============================IMPORT LIBRARIES============================
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import math
import matplotlib.gridspec as gridspec

# Data preprocessing
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Sklearn Library
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)


def save_sklearn_model(model, model_name, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def save_model_and_history(
    model, history, model_name, model_dir="models", history_dir="models/history"
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f"{model_name}.keras")
    model.save(model_path)

    # Save training history
    history_path = os.path.join(history_dir, f"{model_name}_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    print(f"Model saved to {model_path}")
    print(f"Full history saved to {history_path}")


def predict_and_show_samples(
    model,
    df,
    class_names,
    num_samples=5,
    random_select=True,
    target_size=(128, 128),
    figsize=(15, 10),
    rows=None,
    cols=None,
    label_col="label",
    prediction_type="Disease",
):
    """
    Predict and check correctness for images

    model: Trained model
    df: DataFrame with at least 'image_path' and 'label' columns
    class_names: List of class names indexed by label index
    num_samples: Number of images to predict
    random_select: If True, randomly select images; otherwise use top rows
    target_size: Target size to resize images
    figsize: Figure size for the plot
    rows: Number of rows for the grid (calculated automatically if None)
    cols: Number of columns for the grid (calculated automatically if None)
    label_col: Name of the column containing the label in the dataframe
    prediction_type: Type of prediction (e.g., "Disease", "Variety", "Age")
    """
    if isinstance(class_names, tuple):
        class_names = list(class_names)
    
    selected_df = (
        df.sample(n=num_samples).reset_index(drop=True)
        if random_select
        else df.head(num_samples).reset_index(drop=True)
    )

    if rows is None or cols is None:
        cols = min(4, num_samples)
        rows = math.ceil(num_samples / cols)

    # Create figure with grid
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    images = []
    predictions = []
    true_labels = []
    is_correct = []
    confidence_scores = []

    # Process all images and predictions
    for i, row in selected_df.iterrows():
        img_path = row["image_path"]
        true_label_index = (
            row[label_col]
            if isinstance(row[label_col], int)
            else class_names.index(row[label_col])
        )

        # Load & process images
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predictions
        prediction = model.predict(img_batch, verbose=0)
        predicted_index = np.argmax(prediction[0])
        confidence_score = prediction[0][predicted_index]
        correct = predicted_index == true_label_index

        # Store results
        images.append(img_resized)
        predictions.append(class_names[predicted_index])
        true_labels.append(class_names[true_label_index])
        is_correct.append(correct)
        confidence_scores.append(confidence_score)

    # Plot all images in the grid
    for i in range(min(len(images), rows * cols)):
        row_idx = i // cols
        col_idx = i % cols

        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.imshow(images[i])
        ax.axis("off")

        # Create title with prediction info
        title = f"True {prediction_type}: {true_labels[i]}\nPredicted {prediction_type}: {predictions[i]}\nConfidence: {confidence_scores[i]:.2f}"

        # Set title color based on correctness
        title_color = "green" if is_correct[i] else "red"
        ax.set_title(title, color=title_color, fontsize=10)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

    # Print summary
    correct_count = sum(is_correct)
    print(
        f"Correctly predicted {correct_count} out of {len(images)} images ({correct_count/len(images)*100:.1f}%)"
    )


def evaluate_model(model, generator, model_name):
    y_true = generator.classes
    steps = int(np.ceil(generator.samples / generator.batch_size))
    y_pred_probs = model.predict(generator, steps=steps)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"--- {model_name} ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=list(generator.class_indices.keys()),
            zero_division=0,
        )
    )
    print("\n")


def get_metric(history, key_options):
    for key in key_options:
        if key in history.history:
            return history.history[key]
    return None


def plot_history(histories, model_names):
    plt.figure(figsize=(18, 7))

    # Accuracy
    plt.subplot(1, 2, 1)
    for history, name in zip(histories, model_names):
        acc = get_metric(history, ["accuracy", "acc"])
        val_acc = get_metric(history, ["val_accuracy", "val_acc"])
        if acc is None or val_acc is None:
            print(f"Warning: Accuracy not found for {name}. Skipping.")
            continue
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, marker="o", label=f"{name} Train")
        plt.plot(epochs, val_acc, marker="x", linestyle="--", label=f"{name} Val")
        # Annotate best val accuracy
        best_epoch = np.argmax(val_acc)
        best_val = val_acc[best_epoch]
        plt.annotate(
            f"{best_val:.2%}",
            (best_epoch + 1, best_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    plt.title("Model Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    for history, name in zip(histories, model_names):
        loss = get_metric(history, ["loss"])
        val_loss = get_metric(history, ["val_loss"])
        if loss is None or val_loss is None:
            print(f"Warning: Loss not found for {name}. Skipping.")
            continue
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, marker="o", label=f"{name} Train")
        plt.plot(epochs, val_loss, marker="x", linestyle="--", label=f"{name} Val")
        # Annotate lowest val loss
        best_epoch = np.argmin(val_loss)
        best_val = val_loss[best_epoch]
        plt.annotate(
            f"{best_val:.4f}",
            (best_epoch + 1, best_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    plt.title("Model Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_roc_curves(models, model_names, test_generator, num_classes):
    plt.figure(figsize=(12, 8))

    # Get true labels and predictions for each model
    y_true = test_generator.classes
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # Plot ROC curve for each model
    for model, name in zip(models, model_names):
        # Get predictions
        y_pred = model.predict(test_generator)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(10):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot micro-average ROC curve
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f'{name} (AUC = {roc_auc["micro"]:.3f})',
            lw=2,
        )

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], "k--", lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":")
    plt.show()


def plot_learning_curve(
    train_loss, val_loss, train_metric, val_metric, metric_name="Accuracy"
):

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, "r--")
    plt.plot(val_loss, "b--")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend(["train", "val"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(train_metric, "r--")
    plt.plot(val_metric, "b--")
    plt.xlabel("epochs")
    plt.ylabel(metric_name)
    plt.legend(["train", "val"], loc="upper left")

    plt.show()


def analyze_model_layers(model, model_name):
    print(f"\n=== {model_name} Layer Analysis ===")
    print(f"Total number of layers: {len(model.layers)}")

    layer_types = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

    print("\nLayer type distribution:")
    for layer_type, count in layer_types.items():
        print(f"{layer_type}: {count}")

    # Count trainable parameters
    trainable_params = sum(
        [layer.count_params() for layer in model.layers if layer.trainable]
    )
    print(f"\nTotal trainable parameters: {trainable_params:,}")


def load_lambda_histories(history_dir="models/history"):
    histories = {}

    for filename in os.listdir(history_dir):
        if filename.startswith("vgg_") and filename.endswith("lambda_history.pkl"):
            model_name = filename.replace("_history.pkl", "")
            with open(os.path.join(history_dir, filename), "rb") as f:
                histories[model_name] = pickle.load(f)

    return histories


def plot_model_results(model_results, title="Model Comparison", figsize=(12, 8)):
    """
    Plots training and validation accuracies for multiple models side by side for easy comparison.

    Parameters:
    - model_results: Dictionary where keys are model names and values are tuples of (train_accuracy, val_accuracy)
    - title: Title for the plot
    - figsize: Figure size as (width, height)
    """
    # Create a DataFrame for easy plotting with seaborn
    data = []
    for model_name, (train_acc, val_acc) in model_results.items():
        data.append({"Model": model_name, "Accuracy": train_acc, "Type": "Training"})
        data.append({"Model": model_name, "Accuracy": val_acc, "Type": "Validation"})

    # Convert to DataFrame for seaborn
    plot_df = pd.DataFrame(data)

    # Set up the figure
    plt.figure(figsize=figsize)

    # Create the grouped bar chart
    ax = sns.barplot(
        x="Model", y="Accuracy", hue="Type", data=plot_df, palette="coolwarm"
    )

    # Add value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3)

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)  # Leave room for value labels
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="", loc="upper right")
    plt.xticks(rotation=45 if len(model_results) > 3 else 0)

    # Adjust layout
    plt.tight_layout()
    plt.show()
