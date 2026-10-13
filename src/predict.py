import joblib
import pandas as pd

#Load the trained model
model = joblib.load("model.pkl")

#Get student information
study_hours = float(input("Enter study hours:"))
sleep_hours = float(input("Enter sleep hours:"))
attendance = float(input("Enter attendance percentage:"))
previous_score = float(input("Enter previous score:"))

#Create input
student =pd.DataFrame( [[study_hours,sleep_hours,attendance,previous_score]],
columns=["study_hours","sleep_hours","attendance","previous_score"])

#predict
prediction = model.predict(student)

print("\n Predicted Final Score:",prediction[0])