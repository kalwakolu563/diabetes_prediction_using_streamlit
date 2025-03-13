# diabetes_prediction_using_streamlit



Diabetes Detection using Machine Learning and Streamlit
Overview
This project implements a Diabetes Detection System using machine learning models and Streamlit for user interaction. The system allows users to input their medical details and predicts whether they are diabetic or not based on trained ML models.

🔹 Key Features:
✅ Five machine learning models for classification
✅ Automatic selection of the best performing model
✅ Interactive user interface with Streamlit for real-time predictions
✅ Exploratory Data Analysis (EDA) for statistical insights
✅ Support for multiple environments using requirements.txt

Project Structure
📂 diabetes.csv – Contains the dataset used for training and evaluation.
📂 eda.py – Performs exploratory data analysis (EDA) using Matplotlib and Seaborn.
📂 app.py – Runs the Streamlit web application for real-time predictions using a trained model.
📂 model.py – Implements five machine learning models and selects the best one for prediction.
📂 requirements.txt – Lists all the necessary libraries for the project.
📂 readme.txt – Provides steps to set up and activate the environment and install dependencies.

Installation and Setup
1️⃣ Create a Virtual Environment (Recommended)
python -m venv env

2️⃣ Activate the Virtual Environment
On Windows:
.\env\Scripts\activate
On macOS/Linux:
source env/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt


Usage
1️⃣ Running Exploratory Data Analysis (EDA)
To visualize the dataset and analyze feature distributions, run:
python eda.py

2️⃣ Training and Selecting the Best Model
To train the models and identify the best-performing one, run:
python model.py
The script will train Logistic Regression, Random Forest, XGBoost, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN), selecting the best-performing model based on evaluation metrics.

3️⃣ Running the Streamlit Web Application
To launch the web app for real-time diabetes prediction, run:
streamlit run app.py
This will start a web interface where users can input their medical details and receive an instant prediction.

Example Usage
🔹 Example User Input in Streamlit App:
Feature	           Value
Pregnancies	         3
Glucose Level	      120
Blood Pressure	     80
Skin Thickness	     25
Insulin Level	      100
BMI	               28.5
Diabetes Pedigree	  0.5
Age	                 35

<img width="234" alt="image" src="https://github.com/user-attachments/assets/1807b317-ff99-4d59-a67a-92cd9a430991" />

🔹 Predicted Output:
"Based on the input values, the system predicts: Positive for Diabetes."
