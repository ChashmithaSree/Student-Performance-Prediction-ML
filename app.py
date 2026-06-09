import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load data
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['StudyHours', 'Attendance', 'AssignmentsCompleted']]
y = data['Result']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

st.title("🎓 Student Performance Prediction")

study_hours = st.number_input(
    "Study Hours",
    min_value=0,
    max_value=24,
    value=6
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=85
)

assignments = st.number_input(
    "Assignments Completed",
    min_value=0,
    value=7
)

if st.button("Predict Result"):

    input_data = pd.DataFrame(
        [[study_hours, attendance, assignments]],
        columns=[
            'StudyHours',
            'Attendance',
            'AssignmentsCompleted'
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Result: {prediction}")
