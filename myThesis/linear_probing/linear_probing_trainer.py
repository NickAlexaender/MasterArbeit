import os
import json
import time
import warnings
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
)


# Konstanten

LABEL_NAMES = {
    0: "background",
    1: "grau", 
    2: "orange",
    3: "blau",
    4: "grau_orange",
    5: "grau_blau",
    6: "orange_blau",
    7: "all",  # Alle drei Farben
}

RANDOM_SEED = 42
TEST_SIZE = 0.20  # 20% Validierung
TRAIN_SIZE = 0.80  # 80% Training


# Ergebnisse vom Linear Probing

@dataclass
class LinearProbeResults:
    layer: str
    timestamp: str
    n_samples_total: int
    n_samples_train: int
    n_samples_val: int
    n_features: int
    n_classes: int
    label_distribution_train: Dict[str, int]
    label_distribution_val: Dict[str, int]
    train_accuracy: float
    train_balanced_accuracy: float
    val_accuracy: float
    val_balanced_accuracy: float
    val_precision_macro: float
    val_recall_macro: float
    val_f1_macro: float
    val_precision_weighted: float
    val_recall_weighted: float
    val_f1_weighted: float
    val_partial_accuracy: float
    val_strict_accuracy: float
    n_partial_correct: int
    n_exact_correct: int
    confusion_matrix: List[List[int]]
    per_class_metrics: Dict[str, Dict[str, float]]
    training_time_seconds: float
    model_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Daten laden und vorbereiten

def load_features_from_csv(
    csv_path: str,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if verbose:
        print(f"📂 Lade Daten aus: {csv_path}")
        print(f"   ⏳ Lese CSV...")
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"   ✓ CSV geladen!")
    
    names = df["Name"].values.tolist()
    labels = df["Label"].values.astype(np.int32)
    feature_cols = [c for c in df.columns if c.startswith("Gewicht")]
    features = df[feature_cols].values.astype(np.float32)
    
    if verbose:
        print(f"   ✓ {len(names):,} Samples geladen")
        print(f"   ✓ {features.shape[1]} Features pro Sample")
        print(f"   ✓ Label-Verteilung: {dict(pd.Series(labels).value_counts().sort_index())}")
    
    return features, labels, names

# Nun reduzieren wir den Datensatz auf einen Bruchteil

def subsample_data(
    features: np.ndarray,
    labels: np.ndarray,
    names: List[str],
    fraction: float,
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if fraction >= 1.0:
        return features, labels, names
    
    n_total = len(labels)
    n_keep = int(n_total * fraction)
    
    if verbose:
        print(f"\n🎲 Subsample: {fraction*100:.0f}% der Daten ({n_keep:,} von {n_total:,})")
    
    _, features_sub, _, labels_sub, _, idx_sub = train_test_split(
        features, labels, np.arange(len(labels)),
        test_size=fraction,
        random_state=random_state,
        stratify=labels,
    )
    
    names_sub = [names[i] for i in idx_sub]
    
    if verbose:
        print(f"   ✓ Label-Verteilung: {dict(pd.Series(labels_sub).value_counts().sort_index())}")
    
    return features_sub, labels_sub, names_sub

# Für die balanced Datensets erstellen wir eine eigene Logik

def create_balanced_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    names: List[str],
    samples_per_class: int = 10000,
    include_combinations_in_concepts: bool = True,
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    np.random.seed(random_state)
    
    if verbose:
        print(f"\n⚖️ Balanced Dataset erstellen:")
    
    if include_combinations_in_concepts:
        # Grau: Label 1 (grau), 4 (grau+orange), 5 (grau+blau), 7 (alle)
        grau_indices = np.where((labels == 1) | (labels == 4) | (labels == 5) | (labels == 7))[0]
        # Orange: Label 2 (orange), 4 (grau+orange), 6 (orange+blau), 7 (alle)
        orange_indices = np.where((labels == 2) | (labels == 4) | (labels == 6) | (labels == 7))[0]
        # Blau: Label 3 (blau), 5 (grau+blau), 6 (orange+blau), 7 (alle)
        blau_indices = np.where((labels == 3) | (labels == 5) | (labels == 6) | (labels == 7))[0]
        # Background: Nur Label 0
        bg_indices = np.where(labels == 0)[0]
        
        concept_indices = {
            "background": bg_indices,
            "grau": grau_indices,
            "orange": orange_indices,
            "blau": blau_indices,
        }
        
        if verbose:
            print(f"   Kombinationen → zählen zu allen enthaltenen Farben")
            print(f"   Verfügbar: bg={len(bg_indices)}, grau={len(grau_indices)}, orange={len(orange_indices)}, blau={len(blau_indices)}")
    else:
        concept_indices = {}
        for label in range(8):
            idx = np.where(labels == label)[0]
            if len(idx) > 0:
                concept_indices[LABEL_NAMES.get(label, str(label))] = idx
    min_available = min(len(idx) for idx in concept_indices.values())
    if samples_per_class == 0 or samples_per_class > min_available:
        samples_per_class = min_available
        if verbose:
            print(f"   Auto-Samples pro Klasse: {samples_per_class:,} (= kleinste Klasse)")
    else:
        if verbose:
            print(f"   Samples pro Klasse: {samples_per_class:,}")
    
    balanced_indices = []
    new_labels = []
    
    for concept_name, indices in concept_indices.items():
        n_available = len(indices)
        
        if n_available < samples_per_class:
            if verbose:
                print(f"   ⚠️ {concept_name}: nur {n_available:,} verfügbar (nehme alle)")
            selected = indices
        else:
            selected = np.random.choice(indices, size=samples_per_class, replace=False)
        
        balanced_indices.extend(selected)
        concept_label = {"background": 0, "grau": 1, "orange": 2, "blau": 3}.get(concept_name, 0)
        new_labels.extend([concept_label] * len(selected))
    shuffle_order = np.random.permutation(len(balanced_indices))
    balanced_indices = np.array(balanced_indices)[shuffle_order]
    new_labels = np.array(new_labels)[shuffle_order]

    features_bal = features[balanced_indices]
    names_bal = [names[i] for i in balanced_indices]
    
    if verbose:
        print(f"   ✓ Gesamt: {len(new_labels):,} Samples")
        print(f"   ✓ Label-Verteilung: {dict(pd.Series(new_labels).value_counts().sort_index())}")
    
    return features_bal, new_labels, names_bal

# Train/Val Split vorbereiten

def prepare_train_val_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    stratify: bool = True,
    verbose: bool = True,
    min_samples_per_class: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if verbose:
        print(f"\n📊 Erstelle Train/Val Split...")
    
    relevant_classes = {0, 1, 2, 3}
    relevant_mask = np.isin(labels, list(relevant_classes))
    
    if verbose:
        n_removed = (~relevant_mask).sum()
        if n_removed > 0:
            print(f"   ⚠️ {n_removed:,} Samples mit Kombinations-Labels (4-7) entfernt")
    
    features = features[relevant_mask]
    labels = labels[relevant_mask]
    
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    valid_classes = [c for c, cnt in class_counts.items() if cnt >= min_samples_per_class]
    removed_classes = [c for c, cnt in class_counts.items() if cnt < min_samples_per_class]
    
    if removed_classes and verbose:
        for c in removed_classes:
            class_name = LABEL_NAMES.get(c, f"class_{c}")
            print(f"   ⚠️ Klasse '{class_name}' entfernt (nur {class_counts[c]} Sample(s))")
    
    if len(valid_classes) < 2:
        raise ValueError(f"Zu wenige valide Klassen übrig: {valid_classes}. Mindestens 2 benötigt.")
    
    valid_mask = np.isin(labels, valid_classes)
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    indices = np.arange(len(labels))
    
    stratify_labels = labels if stratify else None
    
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        features, labels, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    
    if verbose:
        print(f"\n📊 Train/Val Split ({int((1-test_size)*100)}%/{int(test_size*100)}%):")
        print(f"   Training:   {len(y_train):,} Samples")
        print(f"   Validierung: {len(y_val):,} Samples")
        
        if stratify:
            train_dist = dict(pd.Series(y_train).value_counts().sort_index())
            val_dist = dict(pd.Series(y_val).value_counts().sort_index())
            print(f"   Train-Verteilung: {train_dist}")
            print(f"   Val-Verteilung:   {val_dist}")
    
    return X_train, X_val, y_train, y_val, idx_train, idx_val

# Feature-Standardisierung

def standardize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    
    # Fit nur auf Training
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if verbose:
        print(f"\n🔧 Feature-Standardisierung:")
        print(f"   Mean (train): {X_train.mean():.4f} → {X_train_scaled.mean():.6f}")
        print(f"   Std (train):  {X_train.std():.4f} → {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_val_scaled, scaler

# Modelle trainieren

def train_sgd_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    alpha: float = 0.0001,
    class_weight: Optional[str] = "balanced",
    verbose: bool = True,
):
    if verbose:
        print(f"\n🚀 Training SGD Classifier (log_loss):")
        print(f"   Max Iterationen: {max_iter}")
        print(f"   Alpha (Regularisierung): {alpha}")
        print(f"   Class Weight: {class_weight}")
    
    model = SGDClassifier(
        loss="log_loss",  # Logistic Regression
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-4,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbose=0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
    )
    
    start_time = time.time()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    if verbose:
        print(f"   ✓ Training abgeschlossen in {training_time:.2f}s")
        if hasattr(model, 'n_iter_'):
            print(f"   ✓ Iterationen: {model.n_iter_}")
    
    return model

# Logistische Regression trainieren

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
    verbose: bool = True,
) -> LogisticRegression:
    if verbose:
        print(f"\n🚀 Training Logistische Regression:")
        print(f"   Solver: {solver}")
        print(f"   Max Iterationen: {max_iter}")
        print(f"   C (Regularisierung): {C}")
        print(f"   Class Weight: {class_weight}")
    
    n_classes = len(np.unique(y_train))
    multi_class = "multinomial" if n_classes > 2 else "auto"
    
    model = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        C=C,
        class_weight=class_weight,
        multi_class=multi_class,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbose=0,
    )
    
    start_time = time.time()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    if verbose:
        print(f"   ✓ Training abgeschlossen in {training_time:.2f}s")
        if hasattr(model, 'n_iter_'):
            print(f"   ✓ Iterationen: {model.n_iter_}")
    
    return model

# Partielle Korrektheit für Farbkonzepte

def get_colors_in_label(label: int) -> set:
    color_map = {
        0: set(),                           # background
        1: {"grau"},                        # grau
        2: {"orange"},                      # orange
        3: {"blau"},                        # blau
        4: {"grau", "orange"},              # grau + orange
        5: {"grau", "blau"},                # grau + blau
        6: {"orange", "blau"},              # orange + blau
        7: {"grau", "orange", "blau"},      # alle
    }
    return color_map.get(label, set())

# 

def compute_partial_correctness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    n_total = len(y_true)
    n_exact = 0
    n_partial = 0
    
    details = {
        "exact_matches": 0,
        "partial_overlap": 0,
        "color_subset_predicted": 0,
        "color_superset_predicted": 0,
        "partial_intersection": 0,
        "wrong": 0,
    }
    
    for true, pred in zip(y_true, y_pred):
        true_colors = get_colors_in_label(true)
        pred_colors = get_colors_in_label(pred)
        
        if true == pred:
            n_exact += 1
            n_partial += 1
            details["exact_matches"] += 1
        elif len(true_colors & pred_colors) > 0:
            n_partial += 1
            if pred_colors < true_colors:
                details["color_subset_predicted"] += 1
            elif pred_colors > true_colors:
                details["color_superset_predicted"] += 1
            else:
                details["partial_intersection"] += 1
            details["partial_overlap"] += 1
        else:
            details["wrong"] += 1
    
    partial_accuracy = n_partial / n_total if n_total > 0 else 0.0
    strict_accuracy = n_exact / n_total if n_total > 0 else 0.0
    
    return {
        "partial_accuracy": partial_accuracy,
        "strict_accuracy": strict_accuracy,
        "n_partial_correct": n_partial,
        "n_exact_correct": n_exact,
        "n_total": n_total,
        "details": details,
    }


# Nun müssen wir die Modelle noch evaluieren, um sie vergleichen zu können

def evaluate_model(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "val",
    verbose: bool = True,
) -> Dict[str, Any]:

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    precision_macro = precision_score(y, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, y_pred)
    labels_present = np.unique(np.concatenate([y, y_pred]))
    precision_per_class = precision_score(y, y_pred, average=None, labels=labels_present, zero_division=0)
    recall_per_class = recall_score(y, y_pred, average=None, labels=labels_present, zero_division=0)
    f1_per_class = f1_score(y, y_pred, average=None, labels=labels_present, zero_division=0)
    
    per_class_metrics = {}
    for i, label in enumerate(labels_present):
        label_name = LABEL_NAMES.get(label, f"class_{label}")
        per_class_metrics[label_name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(np.sum(y == label)),
        }
    partial_metrics = compute_partial_correctness(y, y_pred)
    
    if verbose:
        print(f"\n📈 {split_name.capitalize()}-Metriken:")
        print(f"   Accuracy (strict):   {accuracy:.4f}")
        print(f"   Accuracy (partial):  {partial_metrics['partial_accuracy']:.4f}")
        print(f"   Balanced Accuracy:   {balanced_acc:.4f}")
        print(f"   F1 (macro):          {f1_macro:.4f}")
        print(f"   F1 (weighted):       {f1_weighted:.4f}")
        details = partial_metrics['details']
        n_partial_only = partial_metrics['n_partial_correct'] - partial_metrics['n_exact_correct']
        print(f"\n   Partielle Korrektheit:")
        print(f"      Exakt korrekt:     {details['exact_matches']:,}")
        print(f"      Partiell korrekt:  {n_partial_only:,}")
        print(f"        - Teilmenge:     {details['color_subset_predicted']:,}")
        print(f"        - Obermenge:     {details['color_superset_predicted']:,}")
        print(f"        - Überschneidung:{details['partial_intersection']:,}")
        print(f"      Falsch:            {details['wrong']:,}")
        
        print(f"\n   Per-Class:")
        for name, metrics in per_class_metrics.items():
            print(f"      {name:12s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f} (n={metrics['support']:,})")
    
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "partial_accuracy": partial_metrics["partial_accuracy"],
        "strict_accuracy": partial_metrics["strict_accuracy"],
        "n_partial_correct": partial_metrics["n_partial_correct"],
        "n_exact_correct": partial_metrics["n_exact_correct"],
        "partial_details": partial_metrics["details"],
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "predictions": y_pred,
    }


# Vollständiges Linear Probing durchführen

def train_linear_probe(
    csv_path: str,
    output_dir: str,
    layer: str,
    test_size: float = TEST_SIZE,
    max_iter: int = 1000,
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
    save_model: bool = True,
    verbose: bool = True,
    use_sgd: bool = True,
    subsample: Optional[float] = None,
    balanced: bool = False,
    samples_per_class: int = 10000,
    experiment_name: str = "natural",
) -> LinearProbeResults:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if verbose:
        print("=" * 70)
        exp_type = "BALANCED" if balanced else "NATURAL"
        print(f"🔬 LINEAR PROBING - {layer.upper()} ({exp_type})")
        if subsample and subsample < 1.0:
            print(f"   (Subsample: {subsample*100:.0f}% der Daten)")
        if balanced:
            print(f"   (Balanced: {samples_per_class:,} pro Klasse)")
        print("=" * 70)
    features, labels, names = load_features_from_csv(csv_path, verbose=verbose)
    
    if balanced:
        features, labels, names = create_balanced_dataset(
            features, labels, names,
            samples_per_class=samples_per_class,
            include_combinations_in_concepts=True,
            verbose=verbose,
        )
    if subsample is not None and subsample < 1.0:
        features, labels, names = subsample_data(
            features, labels, names, subsample, verbose=verbose
        )

    X_train, X_val, y_train, y_val, idx_train, idx_val = prepare_train_val_split(
        features, labels, test_size=test_size, verbose=verbose
    )

    if verbose:
        print(f"\n🔧 Standardisiere Features...")
    X_train_scaled, X_val_scaled, scaler = standardize_features(X_train, X_val, verbose=verbose)
    start_time = time.time()
    if use_sgd:
        model = train_sgd_classifier(
            X_train_scaled, y_train,
            max_iter=max_iter,
            alpha=1.0 / C,
            class_weight=class_weight,
            verbose=verbose,
        )
    else:
        model = train_logistic_regression(
            X_train_scaled, y_train,
            max_iter=max_iter,
            C=C,
            class_weight=class_weight,
            verbose=verbose,
        )
    training_time = time.time() - start_time
    train_metrics = evaluate_model(model, X_train_scaled, y_train, split_name="train", verbose=verbose)
    val_metrics = evaluate_model(model, X_val_scaled, y_val, split_name="val", verbose=verbose)
    train_dist = {LABEL_NAMES.get(k, str(k)): int(v) for k, v in pd.Series(y_train).value_counts().items()}
    val_dist = {LABEL_NAMES.get(k, str(k)): int(v) for k, v in pd.Series(y_val).value_counts().items()}
    
    results = LinearProbeResults(
        layer=layer,
        timestamp=timestamp,
        n_samples_total=len(labels),
        n_samples_train=len(y_train),
        n_samples_val=len(y_val),
        n_features=features.shape[1],
        n_classes=len(np.unique(labels)),
        label_distribution_train=train_dist,
        label_distribution_val=val_dist,
        train_accuracy=train_metrics["accuracy"],
        train_balanced_accuracy=train_metrics["balanced_accuracy"],
        val_accuracy=val_metrics["accuracy"],
        val_balanced_accuracy=val_metrics["balanced_accuracy"],
        val_precision_macro=val_metrics["precision_macro"],
        val_recall_macro=val_metrics["recall_macro"],
        val_f1_macro=val_metrics["f1_macro"],
        val_precision_weighted=val_metrics["precision_weighted"],
        val_recall_weighted=val_metrics["recall_weighted"],
        val_f1_weighted=val_metrics["f1_weighted"],
        val_partial_accuracy=val_metrics["partial_accuracy"],
        val_strict_accuracy=val_metrics["strict_accuracy"],
        n_partial_correct=val_metrics["n_partial_correct"],
        n_exact_correct=val_metrics["n_exact_correct"],
        confusion_matrix=val_metrics["confusion_matrix"],
        per_class_metrics=val_metrics["per_class_metrics"],
        training_time_seconds=training_time,
    )
    os.makedirs(output_dir, exist_ok=True)
    exp_suffix = f"_{experiment_name}" if experiment_name != "natural" else ""
    results_path = os.path.join(output_dir, f"linear_probe_results_{layer}{exp_suffix}_{timestamp}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n💾 Ergebnisse gespeichert: {results_path}")
    if save_model:
        import joblib
        model_path = os.path.join(output_dir, f"linear_probe_model_{layer}{exp_suffix}_{timestamp}.joblib")
        joblib.dump({"model": model, "scaler": scaler, "label_names": LABEL_NAMES}, model_path)
        results.model_path = model_path
        if verbose:
            print(f"💾 Modell gespeichert: {model_path}")
    if verbose:
        exp_type = "BALANCED" if balanced else "NATURAL"
        print("\n" + "=" * 70)
        print(f"📊 ZUSAMMENFASSUNG ({exp_type})")
        print("=" * 70)
        print(f"   Layer:              {layer}")
        print(f"   Samples (total):    {results.n_samples_total:,}")
        print(f"   Train/Val Split:    {results.n_samples_train:,} / {results.n_samples_val:,}")
        print(f"   Features:           {results.n_features}")
        print(f"   Klassen:            {results.n_classes}")
        print(f"   ---")
        print(f"   Train Accuracy:     {results.train_accuracy:.4f}")
        print(f"   Val Accuracy:       {results.val_accuracy:.4f}")
        print(f"   Val Balanced Acc:   {results.val_balanced_accuracy:.4f}")
        print(f"   Val F1 (macro):     {results.val_f1_macro:.4f}")
        print("=" * 70)
    
    return results

# Wir führen jetzt beide Experiemnte durch: Natural und Balanced

def run_both_experiments(
    csv_path: str,
    output_dir: str,
    layer: str,
    samples_per_class: int = 10000,
    natural_subsample: float = 0.1,
    **kwargs,
) -> Dict[str, LinearProbeResults]:
    kwargs.pop('subsample', None)
    
    results = {}
    # 1. Natural Test (originale Verteilung, 10% random)
    print("\n" + "🌿" * 35)
    print("   EXPERIMENT 1: NATURAL DISTRIBUTION")
    print("🌿" * 35)
    
    results["natural"] = train_linear_probe(
        csv_path=csv_path,
        output_dir=output_dir,
        layer=layer,
        balanced=False,
        experiment_name="natural",
        subsample=natural_subsample,
        **kwargs,
    )
    
    # 2. Balanced Test (10k pro Klasse, kein zusätzliches subsample)
    print("\n" + "⚖️" * 35)
    print("   EXPERIMENT 2: BALANCED DISTRIBUTION")
    print("⚖️" * 35)
    
    results["balanced"] = train_linear_probe(
        csv_path=csv_path,
        output_dir=output_dir,
        layer=layer,
        balanced=True,
        samples_per_class=samples_per_class,
        experiment_name="balanced",
        subsample=None,
        **kwargs,
    )
    print("\n" + "=" * 70)
    print("📊 VERGLEICH: NATURAL vs BALANCED")
    print("=" * 70)
    print(f"{'Metrik':<25} {'Natural':>15} {'Balanced':>15}")
    print("-" * 57)
    print(f"{'Samples (total)':<25} {results['natural'].n_samples_total:>15,} {results['balanced'].n_samples_total:>15,}")
    print(f"{'Klassen':<25} {results['natural'].n_classes:>15} {results['balanced'].n_classes:>15}")
    print("-" * 57)
    print(f"{'Val Accuracy (strict)':<25} {results['natural'].val_strict_accuracy:>15.4f} {results['balanced'].val_strict_accuracy:>15.4f}")
    print(f"{'Val Accuracy (partial)':<25} {results['natural'].val_partial_accuracy:>15.4f} {results['balanced'].val_partial_accuracy:>15.4f}")
    print(f"{'Val Balanced Accuracy':<25} {results['natural'].val_balanced_accuracy:>15.4f} {results['balanced'].val_balanced_accuracy:>15.4f}")
    print(f"{'Val F1 (macro)':<25} {results['natural'].val_f1_macro:>15.4f} {results['balanced'].val_f1_macro:>15.4f}")
    print("=" * 70)
    
    return results

# DAs ganze machen wir jetzt für alle Layer

def train_all_layers(
    base_dir: str,
    output_dir: str,
    layer: Optional[str] = None,
    run_both: bool = False,
    samples_per_class: int = 10000,
    natural_subsample: float = 0.1,
    csv_name: str = "patches.csv",
    **kwargs,
) -> Dict[str, LinearProbeResults]:
    """
    Trainiert Linear Probes für alle (oder ein spezifisches) Layer.
    
    Args:
        base_dir: Verzeichnis mit Layer-Unterordnern (layer0/, layer1/, ...)
        output_dir: Ausgabeverzeichnis
        layer: Optionales spezifisches Layer (z.B. "layer1"), sonst alle
        run_both: True um Natural und Balanced durchzuführen
        samples_per_class: Samples pro Klasse für Balanced
        natural_subsample: Anteil für Natural Experiment (default: 0.1 = 10%)
        csv_name: Name der CSV-Datei (patches.csv für Encoder, queries.csv für Decoder)
        **kwargs: Weitere Parameter für train_linear_probe
        
    Returns:
        Dictionary mit Ergebnissen pro Layer
    """
    results = {}
    
    if layer is not None:
        layers_to_process = [layer]
    else:
        layers_to_process = []
        for name in sorted(os.listdir(base_dir)):
            if name.startswith("layer") and os.path.isdir(os.path.join(base_dir, name)):
                csv_path = os.path.join(base_dir, name, csv_name)
                if os.path.exists(csv_path):
                    layers_to_process.append(name)
    
    print(f"🔍 Gefundene Layer: {layers_to_process}")
    
    for layer_name in layers_to_process:
        csv_path = os.path.join(base_dir, layer_name, csv_name)
        
        if not os.path.exists(csv_path):
            print(f"⚠️ Überspringe {layer_name}: {csv_path} nicht gefunden")
            continue
        
        layer_output_dir = os.path.join(output_dir, layer_name)
        
        try:
            if run_both:
                both_results = run_both_experiments(
                    csv_path=csv_path,
                    output_dir=layer_output_dir,
                    layer=layer_name,
                    samples_per_class=samples_per_class,
                    natural_subsample=natural_subsample,
                    **kwargs,
                )
                results[f"{layer_name}_natural"] = both_results["natural"]
                results[f"{layer_name}_balanced"] = both_results["balanced"]
            else:
                result = train_linear_probe(
                    csv_path=csv_path,
                    output_dir=layer_output_dir,
                    layer=layer_name,
                    **kwargs,
                )
                results[layer_name] = result
        except Exception as e:
            print(f"❌ Fehler bei {layer_name}: {e}")
            import traceback
            traceback.print_exc()
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("📊 GESAMT-ZUSAMMENFASSUNG")
        print("=" * 70)
        print(f"{'Layer':<12} {'Train Acc':>10} {'Val Acc':>10} {'Val Bal Acc':>12} {'Val F1':>10}")
        print("-" * 56)
        for layer_name, res in sorted(results.items()):
            print(f"{layer_name:<12} {res.train_accuracy:>10.4f} {res.val_accuracy:>10.4f} {res.val_balanced_accuracy:>12.4f} {res.val_f1_macro:>10.4f}")
        print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Linear Probing Training")
    parser.add_argument("--base-dir", type=str, required=True, help="Verzeichnis mit Layer-Ordnern")
    parser.add_argument("--output-dir", type=str, required=True, help="Ausgabeverzeichnis")
    parser.add_argument("--layer", type=str, default=None, help="Spezifisches Layer (optional)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validierungsanteil (default: 0.2)")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max Iterationen")
    parser.add_argument("--C", type=float, default=1.0, help="Regularisierung")
    
    args = parser.parse_args()
    
    train_all_layers(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        layer=args.layer,
        test_size=args.test_size,
        max_iter=args.max_iter,
        C=args.C,
    )
