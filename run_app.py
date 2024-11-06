import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(page_title="Obesity Prediction App", page_icon="🍏", layout="wide")

# Sidebar configuration
st.sidebar.header("Navigation")
st.sidebar.markdown("Welcome to the Obesity Prediction App!")
st.sidebar.markdown("""
    Use the options below to navigate through the app:
    - **Home**: Overview of the application.
    - **Predict**: Get predictions for obesity categories.
    - **Contact**: Contact information and links.
""")

# Sidebar navigation
page = st.sidebar.selectbox("Select a page:", ["Home", "Predict", "Contact"])


# Load the model and label encoder
@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


best_model = load_model('trained_models/trained_models.pkl')
label_encoder = load_model('trained_models/label_encoders.pkl')

if page == "Home":
    st.title("Obesity Category Prediction")
    st.markdown("""
        Welcome to the **Obesity Prediction App**! This tool allows you to predict obesity categories based on input data.
        The app will automatically load the test data and generate predictions.
    """)

elif page == "Predict":
    st.title("Predict Obesity Categories")

    # Load test data and prepare for prediction
    test_data = pd.read_csv("datasets/test.csv")  # Automatically load the predefined test CSV
    test_predictions = best_model.predict(test_data)

    prediction_table = pd.DataFrame({
        'ID': test_data['id'],
        'Predicted Obesity Category': label_encoder.inverse_transform(test_predictions)
    })

    # Display prediction results
    st.success("Predictions generated successfully!")
    st.write("Below are the predictions for the test data:")
    st.dataframe(prediction_table)

    # Download option for predictions
    st.markdown("### Download Predictions")
    csv = prediction_table.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="obesity_predictions.csv",
        mime="text/csv",
        help="Click to download predictions as a CSV file"
    )

elif page == "Contact":
    st.title("Contact Information")
    col1, col2 = st.columns([1, 3])  # Create two columns

    with col1:
        st.image("footer_images/image.jpg", width=100)  # Provide the path to your photo

    with col2:
        st.markdown("""
            **Mohamed Naceur Mahmoud**  
            [LinkedIn Profile](https://www.linkedin.com/in/your-profile)  
            📧 Email: [your_email@example.com](mailto:your_email@example.com)  
            📞 Phone: +123456789  
        """)

# Footer
st.markdown("---")
st.markdown("Powered by Machine Learning & Streamlit")
