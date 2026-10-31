"""Patient-balanced neural training with validation-only early stopping."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .dataset import collate_patient_graphs
from .metrics import choose_balanced_threshold, patient_metrics


def _move_batch(batch, device):
    batch["graphs"] = batch["graphs"].to(device)
    batch["roi_to_patient"] = batch["roi_to_patient"].to(device)
    batch["labels"] = batch["labels"].to(device)
    return batch


def _ranking_loss(logits, labels):
    import torch.nn.functional as F

    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    if len(positive) == 0 or len(negative) == 0:
        return logits.new_zeros(())
    return F.softplus(-(positive[:, None] - negative[None, :])).mean()


def predict_model(model, dataset, batch_size: int, device: str):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_patient_graphs,
    )
    model.eval()
    probabilities = []
    labels = []
    patient_ids = []
    with torch.no_grad():
        for batch in loader:
            patient_ids.extend(batch["patient_ids"])
            batch = _move_batch(batch, device)
            logits = model(batch["graphs"], batch["roi_to_patient"])
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch["labels"].cpu().numpy().astype(int).tolist())
    return np.asarray(labels), np.asarray(probabilities), patient_ids


def train_neural_fold(
    model,
    train_dataset,
    validation_dataset,
    test_dataset,
    training_config: Mapping[str, Any],
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    batch_size = int(training_config.get("patient_batch_size", 4))
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_patient_graphs,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 3e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-3)),
    )
    train_labels = np.asarray([patient["label"] for patient in train_dataset.patients])
    positives = max(int(train_labels.sum()), 1)
    negatives = max(int(len(train_labels) - positives), 1)
    pos_weight = torch.tensor(negatives / positives, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
    )

    max_epochs = int(training_config.get("max_epochs", 200))
    patience = int(training_config.get("patience", 25))
    min_epochs = int(training_config.get("min_epochs", 20))
    ranking_weight = float(training_config.get("ranking_loss_weight", 0.1))
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["graphs"], batch["roi_to_patient"])
            loss = criterion(logits, batch["labels"])
            if ranking_weight:
                loss = loss + ranking_weight * _ranking_loss(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        y_val, p_val, _ = predict_model(model, validation_dataset, batch_size, device)
        from sklearn.metrics import roc_auc_score

        val_auc = float(roc_auc_score(y_val, p_val))
        scheduler.step(val_auc)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auc": val_auc})
        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch >= min_epochs and epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    y_val, p_val, val_ids = predict_model(model, validation_dataset, batch_size, device)
    threshold = choose_balanced_threshold(y_val, p_val)
    y_test, p_test, test_ids = predict_model(model, test_dataset, batch_size, device)
    result = {
        "best_epoch": best_epoch,
        "validation": patient_metrics(y_val, p_val, threshold),
        "test": patient_metrics(y_test, p_test, threshold),
        "predictions": [
            {"patient_id": patient_id, "label": int(label), "probability": float(probability)}
            for patient_id, label, probability in zip(test_ids, y_test, p_test)
        ],
        "history": history,
    }
    checkpoint = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "threshold": threshold,
        "best_epoch": best_epoch,
    }
    return result, checkpoint

