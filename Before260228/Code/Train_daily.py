"""
Train_daily.py
LSTM 일별 이진 분류 모델 훈련 전용

예측: 다음날 종가 2% 이상 상승 여부
입력: Data/daily_features.pkl
출력: daily_cls_model.pt  (가중치)
      daily_cls_model_meta.json  (설정·손실·평가)
평가: Test_daily.py에서 별도 수행
"""

import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

CONFIG = {
    "seq_len": 20,
    "pred_horizon": 1,
    "input_size": 58,
    "hidden_size": 256,
    "num_layers": 3,
    "output_size": 2,
    "dropout": 0.1,
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 300,
    "patience": 50,
    "max_grad_norm": 1.0,
    "weight_decay": 1e-4,
    "threshold": 0.02,  # 2%
}

MODEL_NAME = "일별 이진 분류 (다음날 2% 상승 예측)"
SAVE_FILENAME = "daily_cls_model.pt"


if __name__ == "__main__":
    print("=" * 60)
    print(f" {MODEL_NAME}")
    print(f" 디바이스: {DEVICE}")
    print("=" * 60)

    # 데이터 로드
    pkl_path = os.path.join(DATA_DIR, "daily_features.pkl")
    print(f"\n데이터 로드: {pkl_path}")
    with open(pkl_path, "rb") as f:
        features_dict = pickle.load(f)
    print(f"  {len(features_dict)}개 종목")

    # 시퀀스 생성
    print(f"\n  분류 기준: 다음날 수익률 >= {CONFIG['threshold']*100:.1f}%")
    train, valid, test = create_sequences(
        features_dict, CONFIG["seq_len"], CONFIG["pred_horizon"],
        CONFIG["threshold"], use_max_within=False,
    )
    train_X, train_y_ret, train_y_cls = train
    valid_X, valid_y_ret, valid_y_cls = valid
    test_X, test_y_ret, test_y_cls = test

    print(f"  Train: {len(train_X):,} | Valid: {len(valid_X):,} "
          f"| Test: {len(test_X):,}")
    print_class_distribution(train_y_cls, "Train")
    print_class_distribution(test_y_cls, "Test ")

    # DataLoader
    train_loader = DataLoader(
        SequenceDataset(train_X, train_y_cls),
        batch_size=CONFIG["batch_size"], shuffle=True)
    valid_loader = DataLoader(
        SequenceDataset(valid_X, valid_y_cls),
        batch_size=CONFIG["batch_size"])

    # 모델
    compute_class_weights(train_y_cls)
    model = LSTMModel(
        CONFIG["input_size"], CONFIG["hidden_size"],
        CONFIG["num_layers"], CONFIG["output_size"],
        CONFIG["dropout"],
    ).to(DEVICE)
    print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 훈련
    save_path = os.path.join(BASE_DIR, SAVE_FILENAME)
    train_losses, valid_losses = train_classification(
        model, train_loader, valid_loader, CONFIG, None,
        MODEL_NAME, save_path=save_path,
    )

    # 저장
    save_model(model, CONFIG, train_losses, valid_losses, None, save_path)

    # Loss 그래프 저장
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, valid_losses, label="Validation Loss", linewidth=1.5)
    best_epoch = int(np.argmin(valid_losses)) + 1
    ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7,
               label=f"Best Epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{MODEL_NAME} — Train vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(BASE_DIR, "daily_loss_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss 그래프 저장: {plot_path}")

    print("\n" + "=" * 60)
    print(" 훈련 완료!")
    print(f" 모델 저장: {save_path}")
    print(f" Loss 그래프: {plot_path}")
    print(f" 테스트: python Test_daily.py --threshold 0.7")
    print("=" * 60)
