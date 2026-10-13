Student Performance AI

A machine learning project that predicts a student's final score using study hours, sleep hours, attendance percentage, and previous exam score.

Features

- Predict student final scores using Linear Regression
- Evaluate model performance using MAE, RMSE, and R²
- Interactive web interface
- Prediction history saved in CSV
- Dashboard with score statistics and visualizations
- Input validation

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- FastAPI
- HTML, CSS, and JavaScript
- Chart.js

How to Run

1. Install the required Python libraries.
2. Train the model using "src/train.py".
3. Start the web application using:

python -m uvicorn app:app --reload

4. Open "http://127.0.0.1:8000" in your browser.

Project Structure

- "data/" — Student dataset
- "src/train.py" — Model training and evaluation
- "src/predict.py" — Score prediction
- "app.py" — FastAPI application
- "static/index.html" — Web interface
- "model.pkl" — Trained model

Disclaimer

This is an educational machine learning project. Predictions are experimental and should not be treated as reliable assessments of real students.