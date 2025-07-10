# CODECRAFT_ML_02

# 🛍️ Retail Customer Segmentation

A machine learning project that uses **K-Means Clustering** to segment customers of a retail store based on their purchase behavior.  
The project provides an interactive **Streamlit UI** for visualization and automatically saves the trained clustering model using **Scikit-learn**.

---

## 🧩 Problem Statement

Create a **K-Means clustering algorithm** to group customers of a retail store based on their purchase history.  
This segmentation helps the business to:

- Identify high-value customers
- Target personalized marketing campaigns
- Improve customer retention strategies

---

## 📂 Project Structure

CODECRAFT_ML_02/
├── app.py # Streamlit web app for clustering
├── train.py # ML logic: preprocessing, training, saving
├── model/
│ └── kmeans_model.pkl # Trained K-Means model (auto-generated)
├── data/
│ └── Mall_Customers.csv # Dataset (optional - use UI upload instead)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🚀 Features

- ✅ Clusters customers using:
  - Annual Income
  - Spending Score
  - (Or other numeric features)
- ✅ Automatically saves trained model (`kmeans_model.pkl`)
- ✅ Responsive drag-and-drop **file uploader**
- ✅ Interactive **Streamlit** UI with visualized clusters
- ✅ Select number of clusters dynamically using slider
- ✅ Cluster centroids shown visually on scatter plot

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Ujjavalmishr/CODECRAFT_ML_02.git
cd CODECRAFT_ML_02

2. Install dependencies
All dependencies are listed in requirements.txt:
text
Copy
Edit

streamlit
pandas
scikit-learn
seaborn
matplotlib
joblib

pip install -r requirements.txt

3. Run the Streamlit app
bash
Copy code
streamlit run app.py
Visit: http://localhost:0000

4.Uplaod .csv file and click on Run Clustering
 After clicking “Run Clustering”
 Your model will be trained and saved.
 model/Kmeans_model.pkl


📈 Model Details
Algorithm: K-Means Clustering

Input Features: Selectable numeric columns (e.g. income, spending score)

Output: Customer cluster label

Model File: model/kmeans_model.pkl


👨‍💻 Author
Ujjaval Mishra
BTECH CSE(AI) 
ABES Institue Of Technology, Ghaziabad
📧 ujjavalmishra439@gmail.com
🌐 https://github.com/Ujjavalmishr