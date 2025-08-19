import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and data
model = joblib.load(r"D:\mini_project_sport_analytics\tourism_project\linear_regression_model.pkl")
df = pd.read_excel(r"D:\mini_project_sport_analytics\tourism_project\df_for_linear_regression.xlsx")

# 2. Prepare UI
st.title("🎯 Attraction Rating Predictor")

# Dropdowns (UserId excluded, Continent removed)
attraction = st.selectbox("Select Attraction", sorted(df['Attraction'].unique()))
visit_mode = st.selectbox("Select Visit Mode", sorted(df['VisitMode'].unique()))

# 3. Filter the row matching selections
filtered_row = df[
    (df['Attraction'] == attraction) &
    (df['VisitMode'] == visit_mode)
]

# 4. Predict if matching rows found
if not filtered_row.empty:
    # Use average of features if multiple matching rows
    X_input = filtered_row[["User_AvgRating", "category_affinity", "bias_sum"]].mean().to_frame().T
    predicted_rating = model.predict(X_input)[0]
    st.success(f"⭐ Predicted Rating: **{predicted_rating:.2f}**")
else:
    st.warning("⚠️ No matching record found for the selected inputs.")

#---------------------------------------------------------------------------------



# Load the model and data
model = joblib.load(r"D:\mini_project_sport_analytics\tourism_project\visitmode_classifier_minimal.pkl")
df = pd.read_excel(r"D:\mini_project_sport_analytics\tourism_project\df_for_classification.xlsx")

st.title("🧭 Visit Mode Classifier")
st.write("Predict the likely visit mode (e.g., Family, Solo, Business...) based on Attraction Type and Region.")

# Dropdown inputs
bucket = st.selectbox("🎯 Select Attraction Bucket", sorted(df['Attraction_Bucket_3'].dropna().unique()))
region = st.selectbox("📍 Select Region", sorted(df['Region'].dropna().unique()))

# Prepare input for model
input_df = pd.DataFrame([[bucket, region]], columns=['Attraction_Bucket_3', 'Region'])
input_encoded = pd.get_dummies(input_df)

# Align with model input features
for col in model.feature_names_in_:
    if col not in input_encoded.columns:
        input_encoded[col] = 0  # Add missing dummy columns

input_encoded = input_encoded[model.feature_names_in_]  # Ensure correct order

# Predict
prediction = model.predict(input_encoded)[0]
st.success(f"🧳 Predicted Visit Mode: **{prediction}**")


#--------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import joblib

# Load saved objects
model = joblib.load(r"D:\mini_project_sport_analytics\tourism_project\kmeans_cluster_model.pkl")
scaler = joblib.load(r"D:\mini_project_sport_analytics\tourism_project\kmeans_scaler.pkl")
df = pd.read_excel(r"D:\mini_project_sport_analytics\tourism_project\df_with_kmeans_clusters.xlsx")

# Title
st.title("🎯 Attraction Recommender (Cluster-Based)")

# User inputs (CityName removed)
visit_mode = st.selectbox("Visit Mode", sorted(df['VisitMode'].unique()))
region = st.selectbox("Region", sorted(df['Region'].unique()))
country = st.selectbox("Country", sorted(df['Country'].unique()))

# Filter matching rows
filtered_df = df[
    (df['VisitMode'] == visit_mode) &
    (df['Region'] == region) &
    (df['Country'] == country)
]

if not filtered_df.empty:
    # Take the first matching row
    row = filtered_df.iloc[0]

    # Prepare input features (exclude CityName)
    input_features = ['VisitYear', 'VisitMonth', 'VisitMode', 'Attraction', 'Continent', 'Region', 'Country', 'Rating']
    
    # Encode the input
    row_encoded = pd.get_dummies(pd.DataFrame([row[input_features]]))
    full_encoded = pd.get_dummies(df[input_features])
    row_encoded = row_encoded.reindex(columns=full_encoded.columns, fill_value=0)

    # Scale
    row_scaled = scaler.transform(row_encoded)

    # Predict cluster
    cluster = model.predict(row_scaled)[0]
    st.success(f"🔍 You belong to Cluster: **{cluster}**")

    # Recommend top 3 attractions from the same cluster
    cluster_group = df[df['Cluster'] == cluster]
    top_attractions = cluster_group['Attraction'].value_counts().head(3)

    st.subheader("🎡 Top 3 Recommended Attractions")
    for i, (attr, count) in enumerate(top_attractions.items(), 1):
        st.markdown(f"**{i}. {attr}** — visited {count} times")
else:
    st.warning("❗ No matching profile found for your inputs.")
