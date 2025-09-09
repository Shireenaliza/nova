# Grab Nova Score: A Fair Credit Scoring Model

This project develops a machine learning model to generate an alternative credit score, called the **Nova Score**, for Grab partners. The score is based on their performance and reliability within the Grab ecosystem, providing an alternative to traditional, potentially exclusionary, credit metrics.

---

## 🎯 Goal

The primary goal is to create a fair and data-driven credit score that enables Grab partners to access financial services based on their proven track record. The model uses a simulated dataset of driver performance metrics to predict the Nova Score, which is then mapped to a conventional credit score range (300-900).

---

## 🚀 Features

* **Custom Credit Scoring Model**: An XGBoost regression model trained on a simulated dataset.
* **Feature Engineering**: Raw performance metrics are transformed into meaningful features like `Income Stability Index` and `Trip Consistency`.
* **Explainability**: Uses **SHAP** values to explain why a driver received a particular score.
* **Actionable Insights**: Employs **DiCE** counterfactuals to suggest concrete actions drivers can take to improve their score.
* **Interactive UI**: A user-friendly Streamlit application allows drivers to input their data and instantly see their predicted score and improvement suggestions.

---

## 💻 How to Run the Project

### Prerequisites

* Python 3.8+
* Git

### Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### Set up the Environment
Install the required Python packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```

### Prepare the Data
Ensure you have a Dataset.csv file in the root directory. It should contain all the raw features specified in the project's documentation. The train_model.py script will read this file and create a new processed_dataset.csv for reference.

### Train the Model
Run the training script to process the data, train the XGBoost model, and save the model artifacts (.joblib files) for the application to use.

```bash
python train_model.py
```

### Launch the Application
Start the Streamlit web application. This will open the user interface in your web browser.

```bash
streamlit run app.py
```

## 📂 Repository Structure
```bash
├── Dataset.csv             # Raw simulated dataset (not included)
├── train_model.py          # Script for data processing and model training
├── app.py                  # Streamlit application for the UI and predictions
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignores model files and other temporary files
└── README.md               # Project documentation
```
