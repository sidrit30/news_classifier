import streamlit as st
import pandas as pd
import joblib
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
from transformers import pipeline

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="centered"
)


# ---------------------------------------------------------
# MODEL & DATA LOADING (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_nb_model():
    # Get the directory that main.py is in
    base_path = os.path.dirname(__file__)
    
    # Construct paths to the models
    model_path = os.path.join(base_path, 'naive_bayes_model', 'naive_bayes_model.pkl')
    vec_path = os.path.join(base_path, 'naive_bayes_model', 'tfidf_vectorizer.pkl')
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


@st.cache_resource
def load_roberta_model():
    """Loads the roBERTa text-classification pipeline."""
    return pipeline("text-classification", model="sidrit30/roberta_news_classifier")


@st.cache_data
def load_dataset():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'data', 'News_Category_Dataset_v3_compact.csv')
    return pd.read_csv(data_path)


nb_model, nb_vectorizer = load_nb_model()
roberta_pipeline = load_roberta_model()

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a View", ["Manual Prediction", "Model Demonstration (Random)"])

# ---------------------------------------------------------
# VIEW 1: MANUAL PREDICTION
# ---------------------------------------------------------
if app_mode == "Manual Prediction":
    st.title("News Category Classifier")
    st.write("Enter a news headline below and select a model to see its predicted category.")

    headline_input = st.text_area("News Headline", placeholder="e.g., Stock prices fall as interest rates rise...")
    model_choice = st.selectbox("Select Model", ["Naive Bayes", "roBERTa"])

    if st.button("Predict Category"):
        if headline_input.strip() == "":
            st.warning("Please enter a headline first.")
        else:
            with st.spinner(f"Predicting using {model_choice}..."):
                if model_choice == "Naive Bayes":
                    # Logic from predict.py
                    test_vec = nb_vectorizer.transform([headline_input])
                    prediction = nb_model.predict(test_vec)[0]
                    st.success(f"**Predicted Category:** {prediction}")

                else:
                    # Logic from predict.py
                    result = roberta_pipeline(headline_input)
                    prediction = result[0]['label']
                    confidence = result[0]['score']
                    st.success(f"**Predicted Category:** {prediction}")
                    st.info(f"Confidence Score: {confidence:.2%}")

# ---------------------------------------------------------
# VIEW 2: RANDOM HEADLINE DEMONSTRATION
# ---------------------------------------------------------
elif app_mode == "Model Demonstration (Random)":
    st.title("roBERTa Model Demonstration")
    st.write("Testing the best performing model against random entries from the dataset.")

    df = load_dataset()

    if st.button("Get Random Headline & Predict"):
        # Sample a random row from the CSV
        random_row = df.sample(n=1).iloc[0]
        headline = random_row['headline']
        actual_category = random_row['category']

        st.markdown("---")
        st.subheader("Random Headline")
        st.write(f"*{headline}*")

        with st.spinner("roBERTa is thinking..."):
            result = roberta_pipeline(headline)
            predicted_category = result[0]['label']

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Prediction", predicted_category)
            with col2:
                st.metric("Actual Category", actual_category)

            if predicted_category.lower() == actual_category.lower():
                st.balloons()
                st.success("The model predicted correctly!")
            else:

                st.error("The model prediction did not match the actual category.")

