import streamlit as st
import pandas as pd
import numpy as np
import os
import time

# Stats & Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# ML
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from scipy import stats

# DL
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense

# Visualization
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

# ===============================================================
# Streamlit Config
# ===============================================================
st.set_page_config(page_title="Ultimate Weather Dashboard", layout="wide")
st.title("🌤 Ultimate Weather Dashboard — EDA, Forecast & Anomaly Detection")

# ===============================================================
# Load Dataset
# ===============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("seattle-weather.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df["weather_encoded"] = df["weather"].astype("category").cat.codes
    df = df.ffill().bfill()
    return df

df = load_data()
numeric_features = ["temp_max", "temp_min", "wind", "precipitation", "weather_encoded"]

# Map encoded weather labels
weather_map = dict(enumerate(df["weather"].astype("category").cat.categories))
df["weather_label"] = df["weather_encoded"].map(weather_map)

# ===============================================================
# Tabs
# ===============================================================
tabs = st.tabs([
    "Univariate Analysis", "Multivariate Analysis", "Weather Classification",
    "Time Series ACF/PACF", "Clustering", "Forecasting", "Hybrid Model", 
    "Leaderboard", "Anomaly Detection", "Animated Dashboard"
])

# ===============================================================
# 1) Univariate Analysis
# ===============================================================
with tabs[0]:
    st.header("📍 Univariate Analysis for All Numeric Features")
    for feature in numeric_features:
        st.subheader(f"Feature: {feature}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Histogram")
            fig = px.histogram(df, x=feature, nbins=40)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.write("Boxplot")
            fig = px.box(df, y=feature)
            st.plotly_chart(fig, use_container_width=True)
        # Rolling mean
        st.write("30-day Rolling Mean")
        fig = px.line(df[feature].rolling(30).mean())
        st.plotly_chart(fig, use_container_width=True)
        # Seasonal decomposition
        st.write("Seasonal Decomposition")
        try:
            res = seasonal_decompose(df[feature], model="additive", period=365)
            c1, c2 = st.columns(2)
            with c1: st.line_chart(res.trend)
            with c2: st.line_chart(res.seasonal)
            st.line_chart(res.resid)
        except:
            st.info("Seasonal decomposition not possible for this feature.")

# ===============================================================
# 2) Multivariate Analysis
# ===============================================================
with tabs[1]:
    st.header("🧭 Multivariate Analysis")
    st.write("### Correlation Heatmap")
    corr = df[numeric_features].corr()
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Scatter Matrix")
    fig = px.scatter_matrix(df.reset_index(), dimensions=numeric_features, color="weather_label")
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Pairwise Scatter (Temp vs Others)")
    for f in ["wind","precipitation"]:
        fig = px.scatter(df, x=f, y="temp_max", color="weather_label", size="precipitation")
        st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 3) Weather Classification Analysis
# ===============================================================
with tabs[2]:
    st.header("🌦 Weather Classification Analysis")
    st.write("### Weather Stats Across All Numeric Features")
    stats_df = df.groupby("weather_label")[numeric_features[:-1]].agg(['mean','median','std','min','max'])
    st.dataframe(stats_df)

    st.write("### Feature Distribution per Weather Class")
    for feature in numeric_features[:-1]:
        fig = px.box(df, x="weather_label", y=feature, color="weather_label")
        st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 4) Time Series ACF/PACF
# ===============================================================
with tabs[3]:
    st.header("📉 ACF & PACF Plots")
    for f in numeric_features[:-1]:
        st.subheader(f"{f}")
        col1, col2 = st.columns(2)
        fig1, ax1 = plt.subplots()
        plot_acf(df[f], lags=50, ax=ax1)
        with col1: st.pyplot(fig1)
        fig2, ax2 = plt.subplots()
        plot_pacf(df[f], lags=50, ax=ax2)
        with col2: st.pyplot(fig2)

# ===============================================================
# 5) Clustering + PCA
# ===============================================================
with tabs[4]:
    st.header("🧬 KMeans + PCA Clustering")
    scaled = StandardScaler().fit_transform(df[numeric_features])
    df["cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(scaled)
    pc = PCA(n_components=2).fit_transform(scaled)
    df["PC1"], df["PC2"] = pc[:,0], pc[:,1]
    fig = px.scatter(df, x="PC1", y="PC2", color="cluster", hover_data=numeric_features)
    st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 6) Deep Learning Forecast
# ===============================================================
with tabs[5]:
    st.header("🧠 DL Forecast Models")
    SEQ = 30
    series = df["temp_max"].values.reshape(-1,1)
    sc = MinMaxScaler()
    scaled = sc.fit_transform(series)
    def make_seq(data, L=SEQ):
        X, y = [], []
        for i in range(L, len(data)):
            X.append(data[i-L:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    X, y = make_seq(scaled)
    split = int(0.8*len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    def build_model(type):
        m = Sequential()
        if type=="LSTM":
            m.add(LSTM(250, return_sequences=True)); m.add(LSTM(200))
        elif type=="GRU":
            m.add(GRU(250, return_sequences=True)); m.add(GRU(200))
        else:
            m.add(Bidirectional(LSTM(200, return_sequences=True))); m.add(Bidirectional(LSTM(100)))
        m.add(Dense(1)); m.compile(optimizer="adam", loss="mse")
        return m
    os.makedirs("models", exist_ok=True)
    mdl_choice = st.selectbox("Choose DL Model", ["LSTM","GRU","BiLSTM"])
    path = f"models/{mdl_choice}.keras"
    @st.cache_resource
    def get_model(name, path):
        if os.path.exists(path): return load_model(path)
        model = build_model(name)
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)
        model.save(path)
        return model
    model = get_model(mdl_choice, path)
    pred = sc.inverse_transform(model.predict(X_test))
    act = sc.inverse_transform(y_test)
    df_dl = pd.DataFrame({"Actual":act.flatten(),"Predicted":pred.flatten()},
                         index=df.index[-len(pred):])
    st.line_chart(df_dl)

# ===============================================================
# 7) Hybrid Model
# ===============================================================
with tabs[6]:
    st.header("🧪 Hybrid Forecast (ARIMA + LSTM Residual)")
    cut = len(df)-30
    ar_model = ARIMA(df["temp_max"][:cut], order=(5,1,2)).fit()
    ar_fore = ar_model.forecast(30)
    residual = df["temp_max"][-30:] - ar_fore.values
    res_scaled = MinMaxScaler().fit_transform(residual.values.reshape(-1,1))
    XR, yR = make_seq(res_scaled, 7)
    hyb = Sequential([LSTM(16), Dense(1)]); hyb.compile(loss="mse", optimizer="adam")
    hyb.fit(XR, yR, epochs=25, verbose=1)
    res_pred = hyb.predict(XR).flatten() * np.std(residual)
    hyb_final = ar_fore.values[-len(res_pred):] + res_pred
    hyb_df = pd.DataFrame({
        "Actual":df["temp_max"][-len(hyb_final):],
        "Hybrid":hyb_final,
        "ARIMA":ar_fore.values[-len(res_pred):]
    }, index=df.index[-len(hyb_final):])
    st.line_chart(hyb_df)

# ===============================================================
# 8) Leaderboard
# ===============================================================
with tabs[7]:
    st.header("🏆 Model Leaderboard")
    def error(a,b): return mean_absolute_error(a,b), mean_squared_error(a,b)
    mae_ar, mse_ar = error(df["temp_max"][-30:], ar_fore)
    mae_dl, mse_dl = error(act.flatten(), pred.flatten())
    mae_hb, mse_hb = error(df["temp_max"][-len(hyb_final):], hyb_final)
    leader = pd.DataFrame({
        "Model":["ARIMA","Deep Learning","Hybrid"],
        "MAE":[mae_ar, mae_dl, mae_hb],
        "MSE":[mse_ar, mse_dl, mse_hb]
    }).sort_values("MSE")
    st.dataframe(leader, use_container_width=True)
    best_model = leader.iloc[0]["Model"]
    st.success(f"🏅 Best Performing Model: {best_model}")

# ===============================================================
# 9) Anomaly Detection
# ===============================================================
with tabs[8]:
    st.header("🚨 Anomaly Detection")
    anomaly_features = ["temp_max","temp_min","wind","precipitation"]
    df_an = df[anomaly_features].copy()
    def detect_iqr(series):
        Q1,Q3=series.quantile([0.25,0.75]); IQR=Q3-Q1
        return (series<(Q1-1.5*IQR)) | (series>(Q3+1.5*IQR))
    def detect_zscore(series, threshold=3):
        return np.abs(stats.zscore(series))>threshold
    iso = IsolationForest(contamination=0.03, random_state=42)
    iso_pred = iso.fit_predict(df_an)
    df_an["Iso_Anomaly"]=np.where(iso_pred==-1,1,0)
    for col in anomaly_features:
        df_an[f"{col}_IQR"] = detect_iqr(df[col])
        df_an[f"{col}_Z"] = detect_zscore(df[col])
    sumtab = pd.DataFrame({
        "Feature": anomaly_features,
        "IQR Outliers": [df_an[f"{c}_IQR"].sum() for c in anomaly_features],
        "Z-score Outliers":[df_an[f"{c}_Z"].sum() for c in anomaly_features],
        "IsoForest Outliers":[df_an["Iso_Anomaly"].sum()]*len(anomaly_features)
    })
    st.dataframe(sumtab, use_container_width=True)
    feature_chosen = st.selectbox("Choose Feature to visualize anomalies", anomaly_features)
    df_plot = df[[feature_chosen]].copy()
    df_plot["Iso"]=df_an["Iso_Anomaly"]
    scatter = alt.Chart(df_plot.reset_index()).mark_circle(size=60).encode(
        x="date:T", y=f"{feature_chosen}:Q",
        color=alt.condition(alt.datum.Iso==1, alt.value("red"), alt.value("steelblue")),
        tooltip=["date", feature_chosen, "Iso"]
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)
    st.info("🔴 Red = Anomaly, useful for data cleaning & unusual events")

# ===============================================================
# 10) Animated Dashboard
# ===============================================================
with tabs[9]:
    st.header("🚀 Live Animated Dashboard")
    st.info("⚡ Smooth live animation of selected feature with clusters and anomalies.")
    feature = st.selectbox("Select Feature to Animate:", numeric_features, index=0)
    window = st.slider("Window Size for Animation:", 10, 100, 30)
    speed = st.slider("Animation Speed (ms per frame):", 50, 500, 150)
    df_anim = df[[feature,"cluster"]].copy()
    df_anim["Iso_Anomaly"] = df_an["Iso_Anomaly"]
    df_anim.reset_index(inplace=True)
    placeholder = st.empty()
    for i in range(window, len(df_anim)):
        frame = df_anim.iloc[i-window:i]
        scatter = px.scatter(frame, x="date", y=feature,
                             color="cluster", size="Iso_Anomaly",
                             color_continuous_scale="Viridis",
                             title=f"{feature} Animation (Window {window})",
                             labels={"Iso_Anomaly":"Anomaly"})
        placeholder.plotly_chart(scatter, use_container_width=True)
        time.sleep(speed / 1000)
