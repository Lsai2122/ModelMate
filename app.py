import streamlit as st
import pandas as pd

# Load the dataset
data = pd.read_csv('Book1.csv')

# Extract column names from dataset
columns = data.columns.tolist()

# Define metric weights (as per Model Mate conditions)
weights = {col: 1.0 for col in columns if col != 'Model'}  # Default weight = 1.0, customize as needed
weights.update({
    'Outliers': 1.5,
    'Skewness': 1.25,
    'Dimensions': 1.0,
    'Correlation': 1.5,
    'Feature Relations': 1.25,
    'Data Size': 0.75,
    'Data Range': 1.0,
    'Feature Selection Req.': 1.0
})

# Streamlit app UI
st.title('Model Mate - ML Model Recommendation System')

# Sidebar input fields
st.sidebar.header('Input Data Characteristics')
input_data = {}

for col in columns:
    if col != 'Model':
        input_data[col] = st.sidebar.checkbox(col, value=False)

# Prediction logic
if st.sidebar.button('Predict'):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    st.write('### Input Data Characteristics')
    st.write(input_df)

    # Function to compute weighted score
    def calculate_weighted_score(model_row, input_data, weights):
        score = 0
        total_weights = 0
        
        for col, value in input_data.items():
            if col in model_row.index and value:  # Consider only present features
                score += weights.get(col, 1.0) * model_row[col]  # Apply weight
                total_weights += weights.get(col, 1.0)
        
        return score / total_weights if total_weights != 0 else 0

    # Recommend models based on weighted score and rating (4 to -2 scale)
    def recommend_top_models(data, input_data, weights):
        model_scores = {}
        
        for _, row in data.iterrows():
            model_name = row['Model']
            score = calculate_weighted_score(row, input_data, weights)
            model_scores[model_name] = score
        
        # Sort models by weighted score (higher is better)
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_models[:2]  # Return top 2 models

    # Get top model recommendations
    top_models = recommend_top_models(data, input_data, weights)

    # Display recommended models
    st.write('### Top 2 Recommended Models')
    for model, score in top_models:
        st.write(f"**{model}** with a score of {score:.2f}")
else:
    st.write("Click the **Predict** button to see the recommended models.")