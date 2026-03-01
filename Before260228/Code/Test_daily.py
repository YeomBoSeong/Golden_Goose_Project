"""
Test_daily.py
저장된 LSTM 일별 이진 분류 모델 평가

사용법:
  python Test_daily.py                    # 기본 threshold 0.7
  python Test_daily.py --threshold 0.5    # threshold 지정
  python Test_daily.py --threshold 0.3 0.5 0.7 0.9  # 여러 threshold 비교
"""

import argparse
import pickle
import os
import torch
from torch.utils.data import DataLoader
from train_utils import (
    DEVICE, LSTMModel, SequenceDataset, create_sequences,
    print_class_distribution, evaluate_classification, print_cls_metrics,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_PATH = os.path.join(BASE_DIR, "daily_cls_model.pkl")

MODEL_NAME = "일별 이진 분류 (다음날 2% 상승 예측)"


def load_model(model_path):
    """저장된 모델과 config 로드"""
    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)

    config = checkpoint["config"]
    model = LSTMModel(
        config["input_size"], config["hidden_size"],
        config["num_layers"], config["output_size"],
        config.get("dropout", 0.1),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="일별 분류 모델 테스트")
    parser.add_argument(
        "--threshold", type=float, nargs="+", default=None,
        help="상승 예측 확률 threshold. 여러 값 지정 시 비교 테스트. 미지정 시 직접 입력.",
    )
    args = parser.parse_args()

    if args.threshold is None:
        th_input = input("pred_threshold 입력 (여러 값은 공백 구분, 예: 0.3 0.5 0.7 0.9): ").strip()
        args.threshold = [float(x) for x in th_input.split()]

    # 모델 로드
    print("=" * 60)
    print(f" {MODEL_NAME} 테스트")
    print(f" 디바이스: {DEVICE}")
    print(f" 모델: {MODEL_PATH}")
    print(f" 테스트 threshold: {args.threshold}")
    print("=" * 60)

    model, config, checkpoint = load_model(MODEL_PATH)
    print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")

    train_losses = checkpoint.get("train_losses")
    valid_losses = checkpoint.get("valid_losses")

    # 데이터 로드
    pkl_path = os.path.join(DATA_DIR, "daily_features.pkl")
    print(f"\n데이터 로드: {pkl_path}")
    with open(pkl_path, "rb") as f:
        features_dict = pickle.load(f)
    print(f"  {len(features_dict)}개 종목")

    # 시퀀스 생성 (테스트 데이터만 사용)
    print(f"\n  분류 기준: 다음날 수익률 >= {config['threshold']*100:.1f}%")
    _, _, test = create_sequences(
        features_dict, config["seq_len"], config["pred_horizon"],
        config["threshold"], use_max_within=False,
    )
    test_X, test_y_ret, test_y_cls = test
    print(f"  Test: {len(test_X):,}")
    print_class_distribution(test_y_cls, "Test ")

    test_loader = DataLoader(
        SequenceDataset(test_X, test_y_cls),
        batch_size=config["batch_size"])

    # 각 threshold에 대해 평가
    for th in args.threshold:
        print(f"\n{'=' * 60}")
        print(f" pred_threshold = {th}")
        print(f"{'=' * 60}")

        metrics = evaluate_classification(
            model, test_loader, test_y_ret,
            train_losses=train_losses, valid_losses=valid_losses,
            pred_threshold=th,
        )
        print_cls_metrics(metrics, f"{MODEL_NAME} [threshold={th}]")

    # 여러 threshold 비교 요약
    if len(args.threshold) > 1:
        print(f"\n{'=' * 60}")
        print(f" Threshold 비교 요약")
        print(f"{'=' * 60}")
        print(f"  {'threshold':>10} | {'정확도':>8} | {'상승정밀도':>10} | "
              f"{'상승재현율':>10} | {'스프레드':>10} | {'상승예측건수':>12}")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

        for th in args.threshold:
            metrics = evaluate_classification(
                model, test_loader, test_y_ret,
                pred_threshold=th,
            )
            acc = metrics["accuracy"]
            prec = metrics["per_class"]["상승"]["precision"]
            rec = metrics["per_class"]["상승"]["recall"]
            buy_ret = metrics.get("buy_avg_return", 0)
            nobuy_ret = metrics.get("nobuy_avg_return", 0)
            spread = buy_ret - nobuy_ret
            buy_cnt = metrics.get("buy_count", 0)
            print(f"  {th:>10.2f} | {acc:>7.2%} | {prec:>9.2%} | "
                  f"{rec:>9.2%} | {spread:>+9.4f} | {buy_cnt:>11,}")
