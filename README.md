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
Installation and Setup

1. Clone the repository

git clone https://github.com/Justin-x22/Student-performance-ai.git
cd Student-performance-ai

2. Create a virtual environment

python -m venv .venv

3. Install dependencies

Windows (PowerShell):

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

4. Run the application

.\.venv\Scripts\python.exe -m uvicorn app:app --reload

Open http://127.0.0.1:8000 in your browser.

Model Evaluation

The model is evaluated using:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R² Score: Coefficient of determination

The current evaluation results are displayed on the application dashboard. Results depend on the dataset and train/test split.

Future Improvements

- Use a larger, real-world student dataset.
- Compare Linear Regression with other machine learning algorithms.
- Improve model validation and generalization.
- Deploy the application online.

Author

GitHub: "Justin-x22" (https://github.com/Justin-x22)
