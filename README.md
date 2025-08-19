# 🗺️ Tourism Analytics Project

This project applies data science and machine learning techniques to a tourism dataset in order to explore visitor behavior, 
predict ratings, classify travel modes, and cluster attractions. Finally, an interactive **Streamlit application** is built to showcase the results.  

------------------------------------------------------------------------------------------------------------------------------

## 📂 Project Structure

├── data/ # Raw and processed datasets
├── preprocessing.ipynb # Data cleaning & feature engineering
├── ml_linear_regression.ipynb # Regression model to predict ratings
├── ml_classification.ipynb # Classification model (e.g., VisitMode)
├── ml_kmeans_clustering.ipynb # K-Means clustering for segmentation
├── app/
│ └── streamlit_app.py # Streamlit-based interactive application
├── requirements.txt # Python dependencies
└── README.md # Project documentation

-------------------------------------------------------------------------------------------------------------------------------


---

## ⚙️ Workflow

### 1. Preprocessing (`preprocessing.ipynb`)
- Load raw dataset  
- Handle missing values  
- Encode categorical variables  
- Feature engineering (e.g., `VisitMonth_sin`, attraction buckets, user statistics)  
- Export cleaned dataset for downstream tasks  

### 2. Linear Regression (`linear_regression.ipynb`)
- Predict tourist **ratings** of attractions  
- Features: `VisitMode`, `Continent`, `Region`, `User_AvgRating`, `Attraction_Bucket`, etc.  
- Models: Baseline Linear Regression, Polynomial features (tested)  
- Evaluation: R², RMSE  

### 3. Classification (`classification.ipynb`)
- Predict **VisitMode** (Couples, Friends, Family, Solo, Business)  
- Encoded categorical + numeric features  
- Models: Logistic Regression, Random Forest, XGBoost  
- Evaluation: Accuracy, Precision, Recall, F1-score  

### 4. K-Means Clustering (`kmeans_clustering.ipynb`)
- Unsupervised clustering of tourist visits  
- Features: ratings, user activity stats, attraction categories  
- Optimal clusters determined by **Elbow Method**  
- Profiles created for each cluster (dominant VisitMode, attraction type, etc.)  

### 5. Streamlit Application (`app/streamlit_app.py`)
- Interactive dashboard for exploring:  
  - Attraction recommendations by cluster  
  - Filter by region, country, and visit mode  
  - Model predictions integrated into the app  
- Run locally with:  
  ```bash
  streamlit run app/streamlit_app.py
