# =====================================================
# ml_predictor_final.py — Modular Version for Dashboard
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO


# =====================================================
# Define main() so this can be imported by app.py
# =====================================================
def main():
    # -----------------------
    # Page setup
    # -----------------------
    st.markdown("""
        <style>
        body {
            background-color: #0E1117;
            color: #FAFAFA;
            font-family: "Inter", sans-serif;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #FAFAFA;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #262730;
            color: #36CFC9 !important;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚚 NexGen Predictive Delivery Delay Risk — Final Version")

    # -----------------------
    # Upload dataset
    # -----------------------
    uploaded_file = st.file_uploader("📂 Upload CSV dataset (must include 'Delivery_Status')", type=["csv"])

    # Sidebar options
    st.sidebar.header("⚙️ Model Options")
    auto_balance = st.sidebar.checkbox("Auto-balance dataset if imbalanced", value=True)
    balance_threshold = st.sidebar.slider("Balance ratio threshold", 1.5, 3.0, 2.0, step=0.5)
    n_estimators = st.sidebar.slider("RandomForest Trees", 50, 500, 200, step=50)
    max_depth = st.sidebar.slider("Max Depth (0 = None)", 0, 50, 10, step=5)

    if uploaded_file:
        # Load CSV safely
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            st.stop()

        st.success("✅ Dataset loaded successfully!")

        # Safe dataframe conversion for PyArrow compatibility
        safe_df = df_raw.copy()
        for c in safe_df.columns:
            if safe_df[c].dtype == "object":
                safe_df[c] = safe_df[c].astype(str)

        st.write("### 🧾 Dataset Preview")
        st.dataframe(safe_df.head())

        st.write("### 📊 Column Data Types")
        dtype_df = pd.DataFrame({"column": safe_df.columns, "dtype": safe_df.dtypes.astype(str).values})
        st.dataframe(dtype_df)

        # Ensure target column exists
        target_col = "Delivery_Status"
        if target_col not in df_raw.columns:
            st.error("❌ Dataset must contain 'Delivery_Status' column (target).")
            st.stop()

        # Detect date column for line chart
        date_col = None
        for choice in ["Order_Date", "Delivery_Date", "Date"]:
            if choice in df_raw.columns:
                date_col = choice
                break

        # Handle missing values
        for c in df_raw.columns:
            if df_raw[c].isnull().any():
                if df_raw[c].dtype == "object":
                    df_raw[c].fillna(df_raw[c].mode().iloc[0], inplace=True)
                else:
                    df_raw[c].fillna(df_raw[c].median(), inplace=True)

        # Store original categories
        original_categories = {
            c: sorted(df_raw[c].dropna().unique().tolist())
            for c in df_raw.select_dtypes(include=["object"]).columns
        }

        # Encode categoricals
        encoders = {}
        df_enc = df_raw.copy()
        for c in df_enc.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df_enc[c] = le.fit_transform(df_enc[c].astype(str))
            encoders[c] = le

        # Show distribution before balancing
        st.write("### ⚖️ Class Distribution (Before Balancing)")
        counts_before = df_enc[target_col].value_counts().sort_index()
        if target_col in encoders:
            labels_before = encoders[target_col].inverse_transform(counts_before.index)
            st.bar_chart(pd.Series(counts_before.values, index=labels_before))
        else:
            st.bar_chart(counts_before)

        # Auto-balance dataset
        df_model = df_enc.copy()
        if auto_balance:
            vc = df_model[target_col].value_counts()
            if vc.max() / max(vc.min(), 1) >= balance_threshold:
                st.info("🔄 Balancing dataset (upsampling minority classes)...")
                majority_n = vc.max()
                parts = []
                for label, cnt in vc.items():
                    part = df_model[df_model[target_col] == label]
                    if cnt < majority_n:
                        part = resample(part, replace=True, n_samples=majority_n, random_state=42)
                    parts.append(part)
                df_model = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                st.info("✅ Dataset balance acceptable — no balancing needed.")

        # Show after balancing
        st.write("### ⚖️ Class Distribution (After Balancing)")
        counts_after = df_model[target_col].value_counts().sort_index()
        if target_col in encoders:
            labels_after = encoders[target_col].inverse_transform(counts_after.index)
            st.bar_chart(pd.Series(counts_after.values, index=labels_after))
        else:
            st.bar_chart(counts_after)

        # Train model
        X = df_model.drop(columns=[target_col])
        y = df_model[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        rf_params = {"n_estimators": n_estimators, "random_state": 42, "class_weight": "balanced"}
        if max_depth > 0:
            rf_params["max_depth"] = int(max_depth)

        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("🎯 Model Accuracy", f"{acc*100:.2f}%")

        # Classification Report
        st.write("### 🧮 Classification Report")
        if target_col in encoders:
            class_names = encoders[target_col].classes_
            st.text(classification_report(y_test, y_pred, target_names=class_names))
        else:
            st.text(classification_report(y_test, y_pred))

        # Confusion Matrix (smaller size)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(4, 3))
        ticklabels = encoders[target_col].classes_ if target_col in encoders else np.unique(y_test)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ticklabels, yticklabels=ticklabels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig_cm)

        # Feature Importance
        fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        st.write("### 🔍 Feature Importance")
        st.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h", title="Feature Importance"), use_container_width=True)

        # Visualizations
        st.write("### 📈 Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            if date_col:
                try:
                    df_time = df_raw.copy()
                    df_time[date_col] = pd.to_datetime(df_time[date_col], errors="coerce")
                    df_time = df_time.dropna(subset=[date_col])
                    orders_by_date = df_time.groupby(df_time[date_col].dt.date).size().reset_index(name="Orders")
                    fig_line = px.line(orders_by_date, x=date_col, y="Orders", title="📅 Orders Over Time", markers=True)
                    st.plotly_chart(fig_line, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Could not plot time series: {e}")
            else:
                st.info("No date column found for line chart.")

        with col2:
            try:
                st.plotly_chart(px.pie(df_raw, names=target_col, title="🥧 Delivery Status Distribution"), use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not plot pie chart: {e}")

        # Predict for new input
        st.markdown("---")
        st.write("### 🧠 Predict for a New Record")
        input_ui = {}
        for col in X.columns:
            if col in original_categories:
                vals = original_categories[col]
                input_ui[col] = st.selectbox(col, vals)
            else:
                minv, maxv = float(X[col].min()), float(X[col].max())
                step = (maxv - minv) / 100 if (maxv - minv) != 0 else 1.0
                input_ui[col] = st.number_input(col, minv, maxv, float(np.mean([minv, maxv])), step=step)

        if st.button("🚀 Predict Delivery Status"):
            input_df = pd.DataFrame([input_ui])
            for c in input_df.columns:
                if c in encoders:
                    input_df[c] = encoders[c].transform(input_df[c].astype(str))
                else:
                    input_df[c] = pd.to_numeric(input_df[c], errors="coerce").fillna(0)

            pred_encoded = model.predict(input_df)[0]
            decoded_pred = (
                encoders[target_col].inverse_transform([int(pred_encoded)])[0]
                if target_col in encoders
                else str(pred_encoded)
            )

            probs = model.predict_proba(input_df)[0]
            decoded_classes = (
                encoders[target_col].inverse_transform(model.classes_.astype(int))
                if target_col in encoders
                else [str(c) for c in model.classes_]
            )

            st.success(f"**Predicted Delivery_Status:** {decoded_pred}")

            prob_df = pd.DataFrame({
                "Delivery_Status": decoded_classes,
                "Probability (%)": (probs * 100).round(2)
            }).sort_values("Probability (%)", ascending=False)

            st.dataframe(prob_df.style.background_gradient(cmap="RdYlGn", subset=["Probability (%)"]))

            fig_prob = px.bar(prob_df, x="Delivery_Status", y="Probability (%)", color="Probability (%)",
                              color_continuous_scale="RdYlGn", text="Probability (%)",
                              title="Prediction Probabilities")
            st.plotly_chart(fig_prob, use_container_width=True)

            csvbuf = BytesIO()
            prob_df.to_csv(csvbuf, index=False)
            st.download_button("⬇️ Download Prediction CSV", csvbuf.getvalue(),
                               file_name="prediction_probabilities.csv", mime="text/csv")

    else:
        st.info("📤 Upload your dataset to begin.")


# =====================================================
# Allow standalone run
# =====================================================
if __name__ == "__main__":
    main()
