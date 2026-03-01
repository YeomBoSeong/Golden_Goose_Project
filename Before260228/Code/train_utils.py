"""
train_utils.py
LSTM 이진 분류 모델 훈련 공통 유틸리티

모델: LSTM -> Attention -> FC
분류: 상승 여부 이진 분류 (미상승/상승)
손실함수: CrossEntropyLoss
스케줄러: Warmup + CosineAnnealing 또는 ReduceLROnPlateau
"""

import json
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
from tqdm import tqdm

# ─── 디바이스 설정 ────────────────────────────────────────
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")  # WDDM TDR 방지를 위한 조기 초기화
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

CLASS_NAMES = ["미상승", "상승"]


# ═══════════════════════════════════════════════════════════
#  Focal Loss
# ═══════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss — 클래스 불균형 대응.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma: 어려운 샘플에 집중하는 정도. 높을수록 쉬운 샘플 loss 감소. (기본 2.0)
    alpha: 클래스별 가중치 텐서. None이면 가중치 없음.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets_oh = torch.zeros_like(logits)
        targets_oh.scatter_(1, targets.unsqueeze(1), 1.0)

        pt = (probs * targets_oh).sum(dim=1)
        log_pt = (log_probs * targets_oh).sum(dim=1)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        return -(focal_weight * log_pt).mean()


# ═══════════════════════════════════════════════════════════
#  모델 (LSTM -> Attention -> FC)
# ═══════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    """LSTM -> Attention -> FC 이진 분류

    Attention: 마지막 시점의 hidden state를 query로 사용하여
    전체 시퀀스에서 중요한 시점에 가중치를 부여.
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.attn_fc = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Attention: query = 마지막 시점, keys = 전체 시퀀스
        query = out[:, -1:, :]                          # (batch, 1, H)
        keys = torch.tanh(self.attn_fc(out))            # (batch, seq_len, H)
        scores = torch.bmm(query, keys.transpose(1, 2)) # (batch, 1, seq_len)
        weights = torch.softmax(scores, dim=-1)          # (batch, 1, seq_len)
        context = torch.bmm(weights, out).squeeze(1)     # (batch, H)

        return self.fc(context)


# ═══════════════════════════════════════════════════════════
#  데이터셋
# ═══════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    """numpy 배열을 보관하고 __getitem__에서 텐서 변환 (메모리 절약)"""
    def __init__(self, X, y):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


# ═══════════════════════════════════════════════════════════
#  데이터 준비
# ═══════════════════════════════════════════════════════════

def make_binary_labels(values, threshold):
    """수익률 -> 이진 레이블 (0=미상승, 1=상승)"""
    return (values >= threshold).astype(np.int64)


def print_class_distribution(y_cls, split_name):
    counts = np.bincount(y_cls, minlength=2)
    total = len(y_cls)
    parts = [f"{CLASS_NAMES[i]} {counts[i]:,}({counts[i]/total:.1%})"
             for i in range(2)]
    print(f"    {split_name}: {' | '.join(parts)}")


def create_sequences(features_dict, seq_len, pred_horizon, threshold,
                     use_max_within=False):
    """데이터 -> 이진 분류 시퀀스 생성

    use_max_within=False: horizon 마지막 시점의 누적수익률 >= threshold 이면 상승
    use_max_within=True:  horizon 내 최대 누적수익률 >= threshold 이면 상승
    """
    all_train = {"X": [], "y_ret": [], "y_cls": []}
    all_valid = {"X": [], "y_ret": [], "y_cls": []}
    all_test = {"X": [], "y_ret": [], "y_cls": []}

    for ticker, df in tqdm(features_dict.items(), desc="시퀀스 생성", unit="종목"):
        data = df.values.astype(np.float32)
        cols = list(df.columns)
        close_logret_idx = cols.index("close_logret")

        n_samples = len(data) - seq_len - pred_horizon
        if n_samples < 10:
            continue

        X = np.array([data[i:i + seq_len] for i in range(n_samples)])

        if pred_horizon == 1:
            y_ret = np.array([
                data[i + seq_len, close_logret_idx]
                for i in range(n_samples)
            ])
        elif use_max_within:
            # horizon 내 각 시점까지의 누적수익률 중 최대값
            y_ret = np.array([
                np.cumsum(
                    data[i + seq_len:i + seq_len + pred_horizon,
                         close_logret_idx]
                ).max()
                for i in range(n_samples)
            ])
        else:
            y_ret = np.array([
                data[i + seq_len:i + seq_len + pred_horizon,
                     close_logret_idx].sum()
                for i in range(n_samples)
            ])

        y_cls = make_binary_labels(y_ret, threshold)

        n_train = int(n_samples * 0.70)
        n_valid = int(n_samples * 0.15)

        for split, s, e in [
            (all_train, 0, n_train),
            (all_valid, n_train, n_train + n_valid),
            (all_test, n_train + n_valid, n_samples),
        ]:
            split["X"].append(X[s:e])
            split["y_ret"].append(y_ret[s:e])
            split["y_cls"].append(y_cls[s:e])

    result = []
    for split in [all_train, all_valid, all_test]:
        X = np.concatenate(split["X"]).astype(np.float32)
        y_ret = np.concatenate(split["y_ret"]).astype(np.float32)
        y_cls = np.concatenate(split["y_cls"])
        result.append((X, y_ret, y_cls))

    return result


# ═══════════════════════════════════════════════════════════
#  훈련 (실험 4: ReduceLROnPlateau)
# ═══════════════════════════════════════════════════════════

def compute_class_weights(y_cls):
    """클래스 분포만 출력 (가중치 미사용)"""
    counts = np.bincount(y_cls, minlength=2).astype(np.float64)
    total = counts.sum()
    for i, name in enumerate(CLASS_NAMES):
        ratio = counts[i] / total
    return None


def train_classification(model, train_loader, valid_loader, config,
                         class_weights, model_name, save_path=None):
    """이진 분류 모델 훈련. save_path가 주어지면 best 모델을 디스크에 즉시 저장."""
    weight_decay = config.get("weight_decay", 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=weight_decay)

    warmup_epochs = config.get("warmup_epochs", 0)
    if warmup_epochs > 0:
        # Warmup + CosineAnnealing
        total_epochs = config["epochs"]

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Warmup: lr을 0.1배에서 1배까지 선형 증가
                return 0.1 + 0.9 * epoch / warmup_epochs
            # Cosine decay
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        scheduler_type = "warmup_cosine"
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5)
        scheduler_type = "plateau"
    label_smoothing = config.get("label_smoothing", 0.0)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(
            next(model.parameters()).device),
            label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    max_grad_norm = config.get("max_grad_norm", 1.0)

    best_valid_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = config.get("patience", 30)
    train_losses, valid_losses = [], []

    epoch_bar = tqdm(range(1, config["epochs"] + 1),
                     desc=f"{model_name} 훈련", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        batch_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)

        model.eval()
        batch_losses = []
        with torch.no_grad():
            for X_b, y_b in valid_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                loss = criterion(model(X_b), y_b)
                batch_losses.append(loss.item())
        valid_loss = np.mean(batch_losses)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        lr = optimizer.param_groups[0]["lr"]
        epoch_bar.set_postfix_str(
            f"train={train_loss:.6f} valid={valid_loss:.6f} "
            f"best={best_valid_loss:.6f} lr={lr:.1e}"
        )
        if scheduler_type == "plateau":
            scheduler.step(valid_loss)
        else:
            scheduler.step()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
            # best 모델을 디스크에 즉시 저장 (훈련 중단 시 복구 가능)
            if save_path:
                torch.save(best_state, save_path + ".best.pt")
                tqdm.write(
                    f"  [epoch {epoch}] Best model saved "
                    f"(valid_loss={valid_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write(
                    f"  Early stopping at epoch {epoch} "
                    f"(patience={patience})")
                break

    model.load_state_dict(best_state)
    return train_losses, valid_losses


# ═══════════════════════════════════════════════════════════
#  평가
# ═══════════════════════════════════════════════════════════

def evaluate_classification(model, test_loader, test_y_ret=None,
                            train_losses=None, valid_losses=None,
                            pred_threshold=None):
    """이진 분류 평가.

    pred_threshold: None이면 argmax (기본), 값이 있으면 상승 확률 >= threshold 일 때 상승으로 예측.
    """
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(DEVICE)
            logits = model(X_b)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y_b.numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)

    if pred_threshold is not None:
        # softmax 확률 기반 threshold 적용
        from scipy.special import softmax as sp_softmax
        probs = sp_softmax(logits, axis=1)
        preds = (probs[:, 1] >= pred_threshold).astype(np.int64)
    else:
        preds = np.argmax(logits, axis=1)
    accuracy = (preds == targets).mean()

    per_class = {}
    for c in range(2):
        pred_mask = preds == c
        true_mask = targets == c
        tp = ((preds == c) & (targets == c)).sum()
        precision = tp / pred_mask.sum() if pred_mask.sum() > 0 else 0
        recall = tp / true_mask.sum() if true_mask.sum() > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
        per_class[CLASS_NAMES[c]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pred_count": int(pred_mask.sum()),
            "true_count": int(true_mask.sum()),
        }

    majority_ratio = max(np.bincount(targets, minlength=2)) / len(targets)

    result = {
        "accuracy": float(accuracy),
        "majority_baseline": float(majority_ratio),
        "per_class": per_class,
        "n_samples": len(preds),
    }

    # 과적합 진단
    if train_losses and valid_losses:
        best_epoch = np.argmin(valid_losses)
        gap = train_losses[best_epoch] - valid_losses[best_epoch]
        ratio = (abs(gap) / valid_losses[best_epoch]
                 if valid_losses[best_epoch] > 0 else 0)
        result["overfit_info"] = {
            "best_epoch": int(best_epoch + 1),
            "total_epochs": len(train_losses),
            "best_train_loss": float(train_losses[best_epoch]),
            "best_valid_loss": float(valid_losses[best_epoch]),
            "gap_ratio": float(ratio),
        }

    # 예측 분포
    pred_dist = np.bincount(preds, minlength=2) / len(preds)
    true_dist = np.bincount(targets, minlength=2) / len(targets)
    result["pred_distribution"] = {
        CLASS_NAMES[i]: float(pred_dist[i]) for i in range(2)}
    result["true_distribution"] = {
        CLASS_NAMES[i]: float(true_dist[i]) for i in range(2)}

    # 매매 시그널 품질
    if test_y_ret is not None:
        returns = test_y_ret.ravel()
        buy_mask = preds == 1
        nobuy_mask = preds == 0
        result["buy_avg_return"] = (
            float(returns[buy_mask].mean()) if buy_mask.sum() > 0 else 0.0)
        result["nobuy_avg_return"] = (
            float(returns[nobuy_mask].mean()) if nobuy_mask.sum() > 0 else 0.0)
        result["buy_count"] = int(buy_mask.sum())
        result["nobuy_count"] = int(nobuy_mask.sum())

    return result


def print_cls_metrics(metrics, model_name):
    print(f"\n{'─' * 55}")
    print(f" {model_name} 테스트 결과 (n={metrics['n_samples']:,})")
    print(f"{'─' * 55}")

    acc = metrics["accuracy"]
    baseline = metrics.get("majority_baseline", 0)
    print(f"\n  [성능]")
    print(f"    전체 정확도: {acc:.2%}  (다수결 베이스라인: {baseline:.2%})")
    if acc > baseline + 0.05:
        print(f"    -- 베이스라인 대비 +{acc - baseline:.2%} 유의미한 개선")
    elif acc > baseline:
        print(f"    -- 베이스라인 대비 +{acc - baseline:.2%} 미미한 개선")
    else:
        print(f"    -- 베이스라인 이하, 학습 실패")

    print(f"\n  [클래스별 성능]")
    for name in CLASS_NAMES:
        c = metrics["per_class"][name]
        print(f"    {name}: 정밀도 {c['precision']:.2%} "
              f"| 재현율 {c['recall']:.2%} "
              f"| F1 {c['f1']:.2%} "
              f"| 예측 {c['pred_count']:,}건 "
              f"| 실제 {c['true_count']:,}건")

    pd_dist = metrics.get("pred_distribution", {})
    td = metrics.get("true_distribution", {})
    if pd_dist and td:
        print(f"\n  [예측 분포 vs 실제 분포]")
        for name in CLASS_NAMES:
            p, t = pd_dist.get(name, 0), td.get(name, 0)
            bias = ("과다 예측" if p > t + 0.05
                    else ("과소 예측" if p < t - 0.05 else "적절"))
            print(f"    {name}: 예측 {p:.1%} vs 실제 {t:.1%}  -- {bias}")

    if "buy_avg_return" in metrics:
        print(f"\n  [매매 시그널]")
        print(f"    상승 예측 시 평균 수익률: "
              f"{metrics['buy_avg_return']:+.4f} "
              f"({metrics['buy_count']:,}건)")
        print(f"    미상승 예측 시 평균 수익률: "
              f"{metrics['nobuy_avg_return']:+.4f} "
              f"({metrics['nobuy_count']:,}건)")
        spread = metrics["buy_avg_return"] - metrics["nobuy_avg_return"]
        print(f"    상승-미상승 스프레드: {spread:+.4f}")

    oi = metrics.get("overfit_info", {})
    if oi:
        print(f"\n  [훈련 진단]")
        print(f"    Best epoch: {oi['best_epoch']} / {oi['total_epochs']}")
        print(f"    Train loss: {oi['best_train_loss']:.6f}")
        print(f"    Valid loss: {oi['best_valid_loss']:.6f}")
        print(f"    갭 비율: {oi['gap_ratio']:.1%}", end="")
        if oi["best_train_loss"] < oi["best_valid_loss"]:
            if oi["gap_ratio"] > 0.5:
                print("  -- 심한 과적합 (dropout+, hidden-)")
            elif oi["gap_ratio"] > 0.2:
                print("  -- 약한 과적합 (dropout 약간+)")
            else:
                print("  -- 양호")
        else:
            print("  -- 과소적합 가능성 (hidden+, layers+)")

        if oi["best_epoch"] < 30:
            print(f"    [!] Early stop 너무 빠름 -> lr 또는 patience 고려")
        elif oi["best_epoch"] == oi["total_epochs"]:
            print(f"    [!] 최대 epoch 도달 -> epochs 증가 필요")


# ═══════════════════════════════════════════════════════════
#  모델 저장
# ═══════════════════════════════════════════════════════════

def save_model(model, config, train_losses, valid_losses, metrics,
               save_path):
    """모델 저장 (torch.save + JSON 메타로 pickle 코드 실행 취약점 제거).

    - <save_path>.pt   : state_dict만 저장 (torch.load weights_only=True 호환)
    - <save_path>_meta.json: config / 손실 / 평가 메타데이터 (JSON, 안전)

    save_path 확장자는 무시하고 항상 .pt / _meta.json 두 파일로 분리 저장.
    """
    import pathlib
    p = pathlib.Path(save_path)
    pt_path   = p.with_suffix(".pt")
    meta_path = p.with_name(p.stem + "_meta.json")

    # 가중치만 저장 → torch.load(weights_only=True) 로 안전하게 로드 가능
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, pt_path)

    # 설정·손실·평가지표는 JSON으로 저장 (임의 코드 실행 불가)
    meta = {
        "model_type": "binary_classification",
        "config": config,
        "train_losses": [float(x) for x in (train_losses or [])],
        "valid_losses": [float(x) for x in (valid_losses or [])],
        "test_metrics": metrics,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  저장: {pt_path}")
    print(f"  메타: {meta_path}")
