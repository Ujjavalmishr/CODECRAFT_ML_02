# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from train import load_data, preprocess_data, train_kmeans, get_centroids, save_model

st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🛍️ Customer Segmentation with K-Means")

st.markdown("Upload a **CSV** file with customer data (e.g., Mall_Customers.csv)")

uploaded_file = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=False)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.replace('–', '-', regex=False)  # Normalize column names
    st.success("✅ File uploaded successfully!")

    st.subheader("📊 Preview of Data")
    st.dataframe(df.head())

    # features = st.multiselect("Select features for clustering:", df.columns.tolist(), 
    #                           default=['Annual Income (k$)', 'Spending Score (1-100)'])

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    features = st.multiselect("Select features for clustering:", numeric_cols, 
                          default=['Annual Income (k$)', 'Spending Score (1-100)'])


    k = st.slider("Choose number of clusters (k):", 2, 10, value=5)

    if st.button("🚀 Run Clustering"):
        if len(features) < 2:
            st.error("Please select at least two features.")
        else:
            with st.spinner("Training model..."):
                X_scaled, scaler = preprocess_data(df, features)
                clusters, model = train_kmeans(X_scaled, k)
                df['Cluster'] = clusters

                # Save model
                os.makedirs("model", exist_ok=True)
                save_model(model)

                st.success("✅ Clustering completed and model saved as 'model/kmeans_model.pkl'")

                # Display Clustered Data
                st.subheader("📋 Clustered Data")
                st.dataframe(df.head())

                # Cluster Plot
                st.subheader("🖼️ Cluster Visualization")
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['Cluster'], palette='tab10', ax=ax)
                centroids = get_centroids(model, scaler, features)
                ax.scatter(centroids[features[0]], centroids[features[1]],
                           c='black', s=200, marker='X', label='Centroids')
                ax.legend()
                st.pyplot(fig)
else:
    st.info("📂 Please upload a dataset to proceed.")
