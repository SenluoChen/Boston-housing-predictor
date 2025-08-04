import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 載入訓練好的模型
model_log = joblib.load("model_log.pkl")
X = joblib.load("X_columns.pkl")  # 訓練時用的特徵 DataFrame

st.title("🏠 房價預測器")

# 使用者輸入
next_to_river = st.checkbox("是否臨河 (CHAS)")
nr_rooms = st.slider("房間數 (RM)", 1, 10, 6)
students_per_classroom = st.slider("每班學生數 (PTRATIO)", 10, 30, 20)
distance_to_town = st.slider("距離市中心 (DIS)", 1, 12, 5)
pollution_level = st.selectbox("污染程度 (NOX)", ["低", "中", "高"])
poverty_level = st.selectbox("貧困程度 (LSTAT)", ["低", "中", "高"])

if st.button("預測房價"):
    # 1️⃣ 用平均特徵作為基準
    X_encoded = pd.get_dummies(X)
    feature_names = model_log.feature_names_in_
    avg_features = pd.DataFrame([X_encoded.mean()], columns=X_encoded.columns)

    # 2️⃣ 補齊缺少的欄位
    for col in feature_names:
        if col not in avg_features.columns:
            avg_features[col] = 0
    avg_features = avg_features[feature_names]

    # 3️⃣ 設定使用者輸入
    avg_features["CHAS_1"] = 1 if next_to_river else 0
    avg_features["RM"] = nr_rooms
    avg_features["PTRATIO"] = students_per_classroom
    avg_features["DIS"] = distance_to_town

    if pollution_level == "低":
        avg_features["NOX"] = X.NOX.quantile(0.25)
    elif pollution_level == "中":
        avg_features["NOX"] = X.NOX.quantile(0.5)
    else:
        avg_features["NOX"] = X.NOX.quantile(0.75)

    if poverty_level == "低":
        avg_features["LSTAT"] = X.LSTAT.quantile(0.25)
    elif poverty_level == "中":
        avg_features["LSTAT"] = X.LSTAT.quantile(0.5)
    else:
        avg_features["LSTAT"] = X.LSTAT.quantile(0.75)

    # 4️⃣ 預測
    log_price_pred = model_log.predict(avg_features)[0]
    price_pred = np.exp(log_price_pred)
    st.success(f"預測房價: {price_pred:.2f} 美元")
