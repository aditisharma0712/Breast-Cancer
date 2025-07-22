import streamlit as st
import pickle

model = pickle.load(open('model.sav', 'rb'))  
  
def predict_cancer(features):
    prediction = model.predict([features])
    return prediction[0]

def main():
    st.title("Breast Cancer Prediction")
    
    # Input fields for features
    radius_mean = st.number_input("Radius Mean", min_value=0.0, step=0.1)
    texture_mean = st.number_input("Texture Mean", min_value=0.0, step=0.1)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, step=0.1)
    area_mean = st.number_input("Area Mean", min_value=0.0, step=0.1)
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, step=0.01)
    
    if st.button("Predict"):
        features = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]
        result = predict_cancer(features)
        st.success(f"The model predicts: {'Malignant' if result == 1 else 'Benign'}")

if __name__ == "__main__":
    main()