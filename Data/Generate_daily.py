"""
Generate_daily.py
일별 데이터 생성 (일별 + 주별 모델 공용)

출력: daily_features.pkl
"""

from Data_Generation import get_all_tickers, generate_daily_dataset

if __name__ == "__main__":
    print("=" * 60)
    print(" 일별 데이터 생성 (일별/주별 모델 공용)")
    print("=" * 60)

    tickers, names = get_all_tickers()
    generate_daily_dataset(tickers)

    print("\n" + "=" * 60)
    print(" 완료!")
    print("=" * 60)
