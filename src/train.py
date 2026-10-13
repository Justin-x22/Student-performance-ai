from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
#Load the dataset
df=pd.read_csv("data/students.csv")
df.columns = df.columns.str.strip()
print(df.columns.tolist())

#Show the first five 5 rows
print("First 5 students:")
print(df.head())

#Show information about the dataset
print("\nDataset information:")
print(df.info())

#Show basic statistics
print("\nDataset statistics:")
print(df.describe())
print("\n Missing valus:")
print(df.isnull().sum())

print("\n Number of rows:",len(df))
print("Number of columns:",len(df.columns))

import matplotlib.pyplot as plt

#Study hours vs final score
plt.scatter(df["study_hours"], df["final_score"])
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final score")

plt.show()

print("\nStudy hours and final scores:")
print(df[["study_hours","final_score"]])

plt.scatter(df["attendance"],df["final_score"])
plt.xlabel("Attendance")
plt.ylabel("final_score")
plt.title("Attendance vs Final Score")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Features
X = df[["study_hours","sleep_hours","attendance","previous_score"]]

#Target
y = df["final_score"]

#split data
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42)

#Create model
model=LinearRegression()

#Train model
model.fit(X_train,y_train)
print("Model trained succesfully")

joblib.dump(model,"model.pkl")
print("Model saved successfully")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)
}

metrics_path = Path(__file__).resolve().parent.parent / "metrics.pkl"
joblib.dump(metrics, metrics_path)
print("Metrics saved successfully!")

print("Predictions:")
print(y_pred)

mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Mean Absolute Error:",mae)
print("R2 Score:",r2)

#Predict final score for a new student

study_hours = float(input("Enter study hours:"))
sleep_hours = float(input("Enter sleep hours:"))
attendance = float(input("Enter attendance percentage:"))
previous_score = float(input("Enter previous score:"))

new_student = [[study_hours,sleep_hours,attendance,previous_score]]
prediction = model.predict(new_student)

print("Predicted Final Score:",prediction[0])


