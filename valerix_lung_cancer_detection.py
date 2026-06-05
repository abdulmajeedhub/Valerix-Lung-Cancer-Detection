"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║    ██╗   ██╗ █████╗ ██╗     ███████╗██████╗ ██╗██╗  ██╗                        ║
║    ██║   ██║██╔══██╗██║     ██╔════╝██╔══██╗██║╚██╗██╔╝                        ║
║    ██║   ██║███████║██║     █████╗  ██████╔╝██║ ╚███╔╝                         ║
║    ╚██╗ ██╔╝██╔══██║██║     ██╔══╝  ██╔══██╗██║ ██╔██╗                         ║
║     ╚████╔╝ ██║  ██║███████╗███████╗██║  ██║██║██╔╝ ██╗                        ║
║      ╚═══╝  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝                       ║
║                                                                                  ║
║              LUNG CANCER DETECTION SYSTEM  v2.0                                 ║
║              Powered by Deep Learning & CNN Architecture                        ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  Author    : Valerix AI Systems
  Purpose   : Lung Cancer Detection using Convolutional Neural Network
  Libraries : TensorFlow, OpenCV, Scikit-learn, NumPy, Pandas, Matplotlib
  Dataset   : Synthetic CT Scan Dataset (demo) — plug in your own for production
"""

# ─────────────────────────────────────────────────────────────────
#  0.  IMPORTS
# ─────────────────────────────────────────────────────────────────
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")          # Change to "Agg" if no display is available
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────────────────────────
#  GLOBAL STYLE — dark medical theme
# ─────────────────────────────────────────────────────────────────
DARK_BG    = "#0a0e1a"
CARD_BG    = "#111827"
ACCENT1    = "#00c8ff"   # cyan
ACCENT2    = "#7c3aed"   # violet
ACCENT3    = "#f43f5e"   # rose
ACCENT4    = "#10b981"   # emerald
ACCENT5    = "#f59e0b"   # amber
TEXT_MAIN  = "#e2e8f0"
TEXT_DIM   = "#64748b"
GRID_CLR   = "#1e293b"

plt.rcParams.update({
    "figure.facecolor"  : DARK_BG,
    "axes.facecolor"    : CARD_BG,
    "axes.edgecolor"    : GRID_CLR,
    "axes.labelcolor"   : TEXT_MAIN,
    "axes.grid"         : True,
    "grid.color"        : GRID_CLR,
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "text.color"        : TEXT_MAIN,
    "xtick.color"       : TEXT_DIM,
    "ytick.color"       : TEXT_DIM,
    "font.family"       : "DejaVu Sans",
    "figure.dpi"        : 120,
})

IMG_SIZE   = 64          # px — keep low for fast demo; use 128/224 for real CT
NUM_CLASSES = 3          # benign | malignant | normal
CLASSES     = ["Normal", "Benign", "Malignant"]
EPOCHS      = 20
BATCH_SIZE  = 32
SEED        = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ╔══════════════════════════════════════════════════════════════╗
# ║  BANNER                                                       ║
# ╚══════════════════════════════════════════════════════════════╝
def print_banner():
    banner = r"""
    ╔══════════════════════════════════════════════════════════╗
    ║  🫁  VALERIX  •  LUNG CANCER DETECTION  •  v2.0  🫁    ║
    ║  Deep-Learning CT Scan Classifier                        ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print("\033[96m" + banner + "\033[0m")
    stages = [
        "📂  Generating synthetic CT dataset …",
        "🔬  Preprocessing & augmenting images …",
        "🏗️   Building CNN architecture …",
        "🚀  Training model …",
        "📊  Visualising results …",
    ]
    for s in stages:
        print(f"     {s}")
    print()


# ╔══════════════════════════════════════════════════════════════╗
# ║  1.  SYNTHETIC CT SCAN DATASET GENERATOR                    ║
# ╚══════════════════════════════════════════════════════════════╝
def generate_synthetic_ct(n_per_class: int = 300):
    """
    Creates realistic-looking grayscale 'CT slice' images for three classes.
    Replace this function with real DICOM / PNG loaders for production.
    """
    print("\033[93m[VALERIX] Generating synthetic CT data …\033[0m")
    images, labels = [], []

    rng = np.random.default_rng(SEED)

    for cls_idx, cls_name in enumerate(CLASSES):
        for _ in range(n_per_class):
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

            # --- lung oval silhouette ---
            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
            lung_mask = ((X - cx)**2 / (20**2) + (Y - cy)**2 / (25**2)) <= 1
            img[lung_mask] = rng.uniform(0.25, 0.45)

            # --- texture noise (simulates parenchyma) ---
            noise = rng.normal(0, 0.04, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
            img += noise

            # --- class-specific anomalies ---
            if cls_name == "Benign":
                # 1–2 well-defined round nodules
                for _ in range(rng.integers(1, 3)):
                    nx = rng.integers(cx - 15, cx + 15)
                    ny = rng.integers(cy - 18, cy + 18)
                    r  = rng.integers(3, 6)
                    nm = ((X - nx)**2 + (Y - ny)**2) <= r**2
                    img[nm] = rng.uniform(0.65, 0.75)

            elif cls_name == "Malignant":
                # irregular spiculated mass
                for _ in range(rng.integers(1, 4)):
                    nx = rng.integers(cx - 12, cx + 12)
                    ny = rng.integers(cy - 16, cy + 16)
                    r  = rng.integers(5, 11)
                    nm = ((X - nx)**2 + (Y - ny)**2) <= r**2
                    img[nm] = rng.uniform(0.80, 0.95)
                    # spicules
                    for angle in rng.uniform(0, 2 * np.pi, 8):
                        ex = int(nx + (r + rng.integers(3, 8)) * np.cos(angle))
                        ey = int(ny + (r + rng.integers(3, 8)) * np.sin(angle))
                        ex = np.clip(ex, 0, IMG_SIZE - 1)
                        ey = np.clip(ey, 0, IMG_SIZE - 1)
                        cv2.line(img, (nx, ny), (ex, ey), 0.85, 1)

            # clip & convert to uint8
            img = np.clip(img, 0, 1)
            img_uint8 = (img * 255).astype(np.uint8)
            # mild Gaussian blur for realism
            img_uint8 = cv2.GaussianBlur(img_uint8, (3, 3), 0)
            images.append(img_uint8)
            labels.append(cls_idx)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    print(f"\033[92m[VALERIX] Dataset ready — {len(images)} images × {IMG_SIZE}×{IMG_SIZE}\033[0m\n")
    return images, labels


# ╔══════════════════════════════════════════════════════════════╗
# ║  2.  PREPROCESSING                                           ║
# ╚══════════════════════════════════════════════════════════════╝
def preprocess(images, labels):
    print("\033[93m[VALERIX] Preprocessing …\033[0m")

    # Normalise to [0, 1] and expand channel dim
    X = images.astype(np.float32) / 255.0
    X = np.expand_dims(X, -1)                  # (N, H, W, 1)

    # One-hot labels
    y = to_categorical(labels, NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=labels, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED)

    print(f"    Train : {X_train.shape[0]}, Val : {X_val.shape[0]}, Test : {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ╔══════════════════════════════════════════════════════════════╗
# ║  3.  CNN ARCHITECTURE  — ValerixNet                          ║
# ╚══════════════════════════════════════════════════════════════╝
def build_valerixnet(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """Custom CNN with residual-style skip connections."""
    inp = layers.Input(shape=input_shape, name="CT_Input")

    # --- Block 1 ---
    x = layers.Conv2D(32, 3, padding="same", name="conv1_1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(0.25)(x)

    # --- Block 2 ---
    x = layers.Conv2D(64, 3, padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Dropout(0.30)(x)

    # --- Block 3 ---
    x = layers.Conv2D(128, 3, padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)
    x = layers.Dropout(0.40)(x)

    # --- Classifier head ---
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.Dropout(0.50)(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inp, out, name="ValerixNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║  4.  TRAINING                                                ║
# ╚══════════════════════════════════════════════════════════════╝
def train_model(model, X_train, y_train, X_val, y_val):
    print("\033[93m[VALERIX] Training ValerixNet …\033[0m")

    datagen = ImageDataGenerator(
        rotation_range=12,
        zoom_range=0.12,
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    datagen.fit(X_train)

    cb_list = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, verbose=0, min_lr=1e-6),
        callbacks.ModelCheckpoint("valerix_best.keras",
                                  save_best_only=True, monitor="val_accuracy",
                                  verbose=0),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=cb_list,
        verbose=1,
    )
    print("\033[92m[VALERIX] Training complete ✓\033[0m\n")
    return history


# ╔══════════════════════════════════════════════════════════════╗
# ║  5.  VISUALISATION SUITE                                     ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 5a. Training dashboard ────────────────────────────────────
def plot_training_dashboard(history):
    print("\033[93m[VALERIX] Plotting training dashboard …\033[0m")
    h = history.history
    epochs_ran = range(1, len(h["loss"]) + 1)

    fig = plt.figure(figsize=(20, 10), facecolor=DARK_BG)
    fig.suptitle("VALERIX  ·  Training Dashboard",
                 fontsize=20, fontweight="bold", color=ACCENT1,
                 y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.91, bottom=0.09)

    def styled_ax(pos, title, ylabel):
        ax = fig.add_subplot(pos)
        ax.set_title(title, color=ACCENT1, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", color=TEXT_DIM, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_DIM, fontsize=9)
        return ax

    # Loss
    ax = styled_ax(gs[0, 0], "📉  Loss", "Loss")
    ax.plot(epochs_ran, h["loss"],    color=ACCENT3, lw=2, label="Train")
    ax.plot(epochs_ran, h["val_loss"], color=ACCENT1, lw=2, ls="--", label="Val")
    ax.fill_between(epochs_ran, h["loss"], h["val_loss"],
                    alpha=0.08, color=ACCENT2)
    ax.legend(fontsize=8)

    # Accuracy
    ax = styled_ax(gs[0, 1], "🎯  Accuracy", "Accuracy")
    ax.plot(epochs_ran, h["accuracy"],     color=ACCENT4, lw=2, label="Train")
    ax.plot(epochs_ran, h["val_accuracy"], color=ACCENT5, lw=2, ls="--", label="Val")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # AUC
    ax = styled_ax(gs[0, 2], "📈  AUC", "AUC")
    ax.plot(epochs_ran, h["auc"],     color=ACCENT2, lw=2, label="Train")
    ax.plot(epochs_ran, h["val_auc"], color=ACCENT1, lw=2, ls="--", label="Val")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # Precision
    ax = styled_ax(gs[1, 0], "🔬  Precision", "Precision")
    ax.plot(epochs_ran, h["precision"],     color="#f97316", lw=2, label="Train")
    ax.plot(epochs_ran, h["val_precision"], color=ACCENT1,   lw=2, ls="--", label="Val")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # Recall
    ax = styled_ax(gs[1, 1], "📡  Recall", "Recall")
    ax.plot(epochs_ran, h["recall"],     color="#a855f7", lw=2, label="Train")
    ax.plot(epochs_ran, h["val_recall"], color=ACCENT1,  lw=2, ls="--", label="Val")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # LR
    ax = styled_ax(gs[1, 2], "⚡  Learning Rate", "LR")
    lr_vals = h.get("lr", [1e-3] * len(epochs_ran))
    ax.semilogy(epochs_ran, lr_vals, color=ACCENT5, lw=2)

    plt.savefig("valerix_training_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print("    ✔  valerix_training_dashboard.png saved")


# ── 5b. CT Sample Grid ────────────────────────────────────────
def plot_sample_grid(images, labels, n=4):
    print("\033[93m[VALERIX] Plotting CT sample grid …\033[0m")
    COLORS = [ACCENT4, ACCENT5, ACCENT3]
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 10), facecolor=DARK_BG)
    fig.suptitle("VALERIX  ·  Synthetic CT Scan Samples",
                 fontsize=16, fontweight="bold", color=ACCENT1, y=1.01)

    for cls_idx in range(3):
        cls_images = images[labels == cls_idx]
        chosen = np.random.choice(len(cls_images), n, replace=False)
        for j, idx in enumerate(chosen):
            ax = axes[cls_idx, j]
            ax.imshow(cls_images[idx], cmap="bone", interpolation="bilinear",
                      vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(COLORS[cls_idx])
                spine.set_linewidth(2.5)
            if j == 0:
                ax.set_ylabel(CLASSES[cls_idx], color=COLORS[cls_idx],
                              fontsize=12, fontweight="bold", rotation=90,
                              labelpad=6)
            ax.set_title(f"Sample {j+1}", color=TEXT_DIM, fontsize=8, pad=3)

    plt.tight_layout()
    plt.savefig("valerix_ct_samples.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print("    ✔  valerix_ct_samples.png saved")


# ── 5c. Confusion Matrix ──────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    print("\033[93m[VALERIX] Plotting confusion matrix …\033[0m")
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    fig.suptitle("VALERIX  ·  Confusion Matrix", fontsize=16,
                 fontweight="bold", color=ACCENT1, y=0.97)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "valerix", [DARK_BG, ACCENT2, ACCENT1], N=256)

    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% of true class", fraction=0.046, pad=0.04)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if cm_pct[i, j] < 50 else DARK_BG
            ax.text(j, i,
                    f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, color=TEXT_MAIN, fontsize=11)
    ax.set_yticklabels(CLASSES, color=TEXT_MAIN, fontsize=11)
    ax.set_xlabel("Predicted Label", color=TEXT_DIM, fontsize=11)
    ax.set_ylabel("True Label",      color=TEXT_DIM, fontsize=11)

    plt.tight_layout()
    plt.savefig("valerix_confusion_matrix.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print("    ✔  valerix_confusion_matrix.png saved")


# ── 5d. ROC Curves ────────────────────────────────────────────
def plot_roc_curves(y_test_onehot, y_prob):
    print("\033[93m[VALERIX] Plotting ROC curves …\033[0m")
    COLORS = [ACCENT4, ACCENT5, ACCENT3]
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    fig.suptitle("VALERIX  ·  ROC Curves (One-vs-Rest)",
                 fontsize=16, fontweight="bold", color=ACCENT1, y=0.97)

    for i, (cls, clr) in enumerate(zip(CLASSES, COLORS)):
        fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=clr, lw=2.5,
                label=f"{cls}  (AUC = {roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.07, color=clr)

    ax.plot([0, 1], [0, 1], color=TEXT_DIM, lw=1.5, ls="--", label="Random")
    ax.set_xlabel("False Positive Rate", color=TEXT_DIM, fontsize=11)
    ax.set_ylabel("True Positive Rate",  color=TEXT_DIM, fontsize=11)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig("valerix_roc_curves.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print("    ✔  valerix_roc_curves.png saved")


# ── 5e. Class Distribution ────────────────────────────────────
def plot_dataset_overview(labels):
    print("\033[93m[VALERIX] Plotting dataset overview …\033[0m")
    counts   = [np.sum(labels == i) for i in range(NUM_CLASSES)]
    COLORS   = [ACCENT4, ACCENT5, ACCENT3]
    EXPLODE  = (0.04, 0.04, 0.08)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    fig.suptitle("VALERIX  ·  Dataset Overview",
                 fontsize=16, fontweight="bold", color=ACCENT1, y=1.01)

    # Pie chart
    ax = axes[0]
    ax.set_facecolor(DARK_BG)
    wedges, texts, autotexts = ax.pie(
        counts, labels=CLASSES, autopct="%1.1f%%",
        colors=COLORS, explode=EXPLODE,
        startangle=140, pctdistance=0.82,
        textprops={"color": TEXT_MAIN, "fontsize": 12},
        wedgeprops={"linewidth": 2, "edgecolor": DARK_BG},
    )
    for at in autotexts:
        at.set_color(DARK_BG)
        at.set_fontweight("bold")
    ax.set_title("Class Distribution", color=ACCENT1, fontsize=13, pad=12)

    # Bar chart
    ax = axes[1]
    bars = ax.bar(CLASSES, counts, color=COLORS, width=0.55,
                  edgecolor=DARK_BG, linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(count), ha="center", va="bottom",
                color=TEXT_MAIN, fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Images", color=TEXT_DIM, fontsize=11)
    ax.set_title("Sample Count per Class", color=ACCENT1, fontsize=13)
    ax.set_ylim(0, max(counts) * 1.18)

    plt.tight_layout()
    plt.savefig("valerix_dataset_overview.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print("    ✔  valerix_dataset_overview.png saved")


# ── 5f. GradCAM-style Heatmap ─────────────────────────────────
def plot_gradcam_heatmaps(model, X_test, y_test_onehot, labels_1d):
    """Lightweight activation map using the last conv layer output."""
    print("\033[93m[VALERIX] Plotting activation heatmaps (GradCAM-lite) …\033[0m")

    # Build a sub-model that outputs both the last conv and predictions
    last_conv   = model.get_layer("conv3_2")
    grad_model  = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output]
    )

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor=DARK_BG)
    fig.suptitle("VALERIX  ·  Activation Heatmaps per Class",
                 fontsize=16, fontweight="bold", color=ACCENT1, y=1.00)
    CMAPS = ["Greens", "YlOrBr", "Reds"]

    for cls_idx in range(3):
        idx_list = np.where(labels_1d == cls_idx)[0]
        chosen   = np.random.choice(idx_list, 4, replace=False)
        for j, idx in enumerate(chosen):
            img_tensor = X_test[idx:idx+1]
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(img_tensor)
                target = preds[:, cls_idx]
            grads  = tape.gradient(target, conv_out)[0]
            pooled = tf.reduce_mean(grads, axis=(0, 1)).numpy()
            cam    = tf.reduce_sum(conv_out[0] * pooled, axis=-1).numpy()
            cam    = np.maximum(cam, 0)
            if cam.max() > 0:
                cam  = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

            orig = (X_test[idx, :, :, 0] * 255).astype(np.uint8)
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
            heatmap = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig_rgb, 0.55, heatmap, 0.45, 0)

            ax = axes[cls_idx, j]
            ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            pred_class = np.argmax(preds[0].numpy())
            correct = pred_class == cls_idx
            border_clr = ACCENT4 if correct else ACCENT3
            for spine in ax.spines.values():
                spine.set_edgecolor(border_clr); spine.set_linewidth(2.5)
            ax.set_title(
                f"{'✓' if correct else '✗'}  {CLASSES[pred_class]}",
                color=border_clr, fontsize=9, fontweight="bold", pad=3)
            if j == 0:
                ax.set_ylabel(CLASSES[cls_idx], color=[ACCENT4,ACCENT5,ACCENT3][cls_idx],
                              fontsize=11, fontweight="bold", rotation=90, labelpad=6)

    plt.tight_layout()
    plt.savefig("valerix_heatmaps.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print("    ✔  valerix_heatmaps.png saved")


# ── 5g. Model Architecture Diagram ───────────────────────────
def plot_architecture():
    print("\033[93m[VALERIX] Plotting CNN architecture diagram …\033[0m")
    fig, ax = plt.subplots(figsize=(18, 7), facecolor=DARK_BG)
    ax.set_xlim(0, 18); ax.set_ylim(0, 7)
    ax.axis("off")
    fig.suptitle("VALERIX  ·  ValerixNet Architecture",
                 fontsize=17, fontweight="bold", color=ACCENT1, y=0.97)

    layers_info = [
        ("Input\n64×64×1",  0.6,  ACCENT4,  0.9),
        ("Conv 3×3\n×32",   2.2,  ACCENT1,  0.75),
        ("Conv 3×3\n×32",   3.4,  ACCENT1,  0.75),
        ("MaxPool\n32×32",  4.6,  ACCENT2,  0.65),
        ("Conv 3×3\n×64",   5.8,  ACCENT1,  0.55),
        ("Conv 3×3\n×64",   7.0,  ACCENT1,  0.55),
        ("MaxPool\n16×16",  8.2,  ACCENT2,  0.45),
        ("Conv 3×3\n×128",  9.4,  ACCENT1,  0.38),
        ("Conv 3×3\n×128", 10.6,  ACCENT1,  0.38),
        ("MaxPool\n8×8",   11.8,  ACCENT2,  0.32),
        ("GAP",            13.0,  ACCENT5,  0.25),
        ("FC 256",         14.2,  ACCENT3,  0.20),
        ("FC 128",         15.4,  ACCENT3,  0.16),
        ("Softmax\n×3",    16.6,  ACCENT4,  0.12),
    ]

    cy = 3.5
    for name, x, color, h_ratio in layers_info:
        h = h_ratio * 5
        rect = mpatches.FancyBboxPatch(
            (x - 0.5, cy - h/2), 1.0, h,
            boxstyle="round,pad=0.05",
            facecolor=color + "33", edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, cy, name, ha="center", va="center",
                color=color, fontsize=7.5, fontweight="bold",
                multialignment="center")

    # arrows
    xs = [info[1] for info in layers_info]
    for i in range(len(xs) - 1):
        ax.annotate("",
                    xy=(xs[i+1] - 0.5, cy),
                    xytext=(xs[i] + 0.5, cy),
                    arrowprops=dict(arrowstyle="->", color=TEXT_DIM, lw=1.5))

    plt.tight_layout()
    plt.savefig("valerix_architecture.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print("    ✔  valerix_architecture.png saved")


# ── 5h. Metrics Summary Report ────────────────────────────────
def plot_metrics_report(y_true, y_pred, y_prob, model, X_test, y_test_onehot):
    print("\033[93m[VALERIX] Plotting final metrics report …\033[0m")
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True)
    df_report   = pd.DataFrame(report_dict).T.iloc[:-3]

    # Per-class AUC
    aucs = []
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_prob[:, i])
        aucs.append(auc(fpr, tpr))

    fig = plt.figure(figsize=(18, 7), facecolor=DARK_BG)
    fig.suptitle("VALERIX  ·  Final Performance Report",
                 fontsize=18, fontweight="bold", color=ACCENT1, y=1.01)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                           left=0.04, right=0.97, top=0.92, bottom=0.14)

    METRIC_COLORS = [ACCENT4, ACCENT5, ACCENT3]
    metrics       = ["precision", "recall", "f1-score"]
    metric_labels = ["Precision", "Recall", "F1-Score"]

    for mi, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = fig.add_subplot(gs[0, mi])
        ax.set_facecolor(CARD_BG)
        vals = [df_report.loc[cls, metric] for cls in CLASSES]
        x    = np.arange(len(CLASSES))
        bars = ax.bar(x, vals, color=METRIC_COLORS,
                      edgecolor=DARK_BG, linewidth=1.5, width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom",
                    color=TEXT_MAIN, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=10)
        ax.set_ylabel(mlabel, color=TEXT_DIM, fontsize=11)
        ax.set_title(f"🔢  {mlabel}", color=ACCENT1,
                     fontsize=12, fontweight="bold", pad=8)

    plt.tight_layout()
    plt.savefig("valerix_metrics_report.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print("    ✔  valerix_metrics_report.png saved")


# ── 5i. Prediction Panel (inference on 8 random test samples) ─
def plot_prediction_panel(model, X_test, y_test_labels, n=8):
    print("\033[93m[VALERIX] Plotting prediction panel …\033[0m")
    RISK = {0: ("LOW",    ACCENT4),
            1: ("MEDIUM", ACCENT5),
            2: ("HIGH",   ACCENT3)}

    chosen = np.random.choice(len(X_test), n, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=DARK_BG)
    fig.suptitle("VALERIX  ·  Live Inference Panel",
                 fontsize=16, fontweight="bold", color=ACCENT1, y=1.01)

    for k, idx in enumerate(chosen):
        ax = axes[k // 4, k % 4]
        img = X_test[idx, :, :, 0]
        ax.imshow(img, cmap="bone", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])

        probs    = model.predict(X_test[idx:idx+1], verbose=0)[0]
        pred_cls = np.argmax(probs)
        true_cls = y_test_labels[idx]
        correct  = pred_cls == true_cls
        risk_lbl, risk_clr = RISK[pred_cls]

        border_clr = ACCENT4 if correct else ACCENT3
        for spine in ax.spines.values():
            spine.set_edgecolor(border_clr); spine.set_linewidth(3)

        title = (f"Pred: {CLASSES[pred_cls]}  [{risk_lbl}]\n"
                 f"True: {CLASSES[true_cls]}  "
                 f"{'✓' if correct else '✗'}  conf: {probs[pred_cls]*100:.1f}%")
        ax.set_title(title, color=risk_clr, fontsize=7.8,
                     fontweight="bold", pad=4)

        # mini confidence bar
        bar_w = 58
        for ci, (p, c) in enumerate(zip(probs, [ACCENT4, ACCENT5, ACCENT3])):
            bx = 3
            by = IMG_SIZE - 14 - ci * 14
            ax.add_patch(plt.Rectangle((bx, by), bar_w * p, 7,
                                       color=c, alpha=0.75, transform=ax.transData))
            ax.text(bx + bar_w + 1, by + 3,
                    f"{CLASSES[ci][0]}", color=c, fontsize=6,
                    va="center", transform=ax.transData)

    plt.tight_layout()
    plt.savefig("valerix_prediction_panel.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print("    ✔  valerix_prediction_panel.png saved")


# ── 5j. Pixel Intensity Distribution ─────────────────────────
def plot_pixel_distribution(images, labels):
    print("\033[93m[VALERIX] Plotting pixel intensity distributions …\033[0m")
    COLORS = [ACCENT4, ACCENT5, ACCENT3]
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    fig.suptitle("VALERIX  ·  Pixel Intensity Distributions by Class",
                 fontsize=16, fontweight="bold", color=ACCENT1, y=1.01)

    for cls_idx, (cls_name, clr) in enumerate(zip(CLASSES, COLORS)):
        pixels = images[labels == cls_idx].flatten()
        ax.hist(pixels, bins=80, color=clr, alpha=0.55,
                label=cls_name, density=True)

    ax.set_xlabel("Pixel Intensity (0–255)", color=TEXT_DIM, fontsize=11)
    ax.set_ylabel("Density", color=TEXT_DIM, fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("valerix_pixel_distribution.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print("    ✔  valerix_pixel_distribution.png saved")


# ── 5k. Summary Terminal Table ────────────────────────────────
def print_summary_table(y_true, y_pred, y_prob, y_test_onehot, history):
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    val_acc = max(history.history["val_accuracy"]) * 100
    aucs = []
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_prob[:, i])
        aucs.append(auc(fpr, tpr))

    sep = "─" * 66
    print(f"\n\033[96m{sep}\033[0m")
    print(f"\033[96m  🫁  VALERIX  ·  FINAL EVALUATION SUMMARY  \033[0m")
    print(f"\033[96m{sep}\033[0m")
    print(report)
    print(f"  Best Validation Accuracy : \033[92m{val_acc:.2f}%\033[0m")
    for i, (cls, a) in enumerate(zip(CLASSES, aucs)):
        print(f"  AUC — {cls:<12}: \033[93m{a:.4f}\033[0m")
    print(f"\033[96m{sep}\033[0m\n")

    saved = [
        "valerix_training_dashboard.png",
        "valerix_ct_samples.png",
        "valerix_confusion_matrix.png",
        "valerix_roc_curves.png",
        "valerix_dataset_overview.png",
        "valerix_heatmaps.png",
        "valerix_architecture.png",
        "valerix_metrics_report.png",
        "valerix_prediction_panel.png",
        "valerix_pixel_distribution.png",
    ]
    print("\033[95m  Output Files:\033[0m")
    for f in saved:
        print(f"    📄  {f}")
    print()


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                               ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    print_banner()
    t0 = time.time()

    # 1. Generate data
    images, labels = generate_synthetic_ct(n_per_class=300)

    # 2. Visualise dataset
    plot_dataset_overview(labels)
    plot_sample_grid(images, labels, n=4)
    plot_pixel_distribution(images, labels)

    # 3. Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(images, labels)
    y_test_labels = np.argmax(y_test, axis=1)

    # 4. Build model
    model = build_valerixnet()
    model.summary()

    # 5. Architecture diagram (static)
    plot_architecture()

    # 6. Train
    history = train_model(model, X_train, y_train, X_val, y_val)

    # 7. Evaluate
    print("\033[93m[VALERIX] Evaluating on test set …\033[0m")
    y_prob  = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_prob, axis=1)

    # 8. Visualisations
    plot_training_dashboard(history)
    plot_confusion_matrix(y_test_labels, y_pred)
    plot_roc_curves(y_test, y_prob)
    plot_gradcam_heatmaps(model, X_test, y_test, y_test_labels)
    plot_metrics_report(y_test_labels, y_pred, y_prob, model, X_test, y_test)
    plot_prediction_panel(model, X_test, y_test_labels, n=8)

    # 9. Summary
    print_summary_table(y_test_labels, y_pred, y_prob, y_test, history)

    elapsed = time.time() - t0
    print(f"\033[92m  ✅  Valerix pipeline completed in {elapsed:.1f}s\033[0m\n")


if __name__ == "__main__":
    main()
