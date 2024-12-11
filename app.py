import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained logistic regression model and scaler
with open('titanic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [1 if sex == 'Female' else 0],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [0 if embarked == 'S' else 1 if embarked == 'C' else 2]
    })
    return data

# Streamlit app layout
def main():
    st.title("Titanic Survival Prediction")
    st.write("Enter passenger details to predict survival probability.")

    # User inputs for prediction
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.slider("Age", 0, 100, 30)
    sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
    parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = st.number_input("Fare", min_value=0.0, value=32.2)
    embarked = st.selectbox("Port of Embarkation (Embarked)", ["S", "C", "Q"])

    # Predict button
    if st.button("Predict"):
    input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"The passenger is predicted to survive with a probability of {prediction_proba:.2%}.")
    else:
        st.error(f"The passenger is predicted not to survive with a probability of {(1 - prediction_proba):.2%}.")


if __name__ == "__main__":
    main()
