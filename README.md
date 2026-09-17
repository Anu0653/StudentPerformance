# Student Performance Analysis and Prediction using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based system that predicts a student's final academic grade (**G3**) using demographic, social, academic, and behavioral factors.

The project follows a complete Machine Learning workflow, starting from data analysis and preprocessing to model training, evaluation, model persistence, and deployment using Streamlit.

The system is designed as an **early performance prediction** tool by excluding the previous-period grades **G1 and G2** from the input features.

---

## 🎯 Objectives

* Analyze factors that may be associated with student performance.
* Perform data quality checks and exploratory data analysis.
* Preprocess categorical and numerical features.
* Train and compare Machine Learning regression models.
* Predict a student's final grade.
* Provide an interactive web interface using Streamlit.

---

## 📊 Dataset

The dataset contains:

* **395 student records**
* **33 columns**
* **30 input features** used for prediction
* **G3** as the target variable

### Target Variable

**G3 — Final Grade**

The target represents the student's final grade on a scale of **0–20**.

### Feature Selection

`G1` and `G2` were excluded because they represent previous-period grades. Excluding them makes the project an early performance prediction system based on other student-related factors.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Quality Check
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature & Target Selection
   ↓
One-Hot Encoding
   ↓
Train/Test Split
   ↓
Linear Regression (Baseline)
   ↓
Random Forest Regression
   ↓
Model Evaluation
   ↓
Save Model using Joblib
   ↓
Streamlit Application
   ↓
Predicted Final Grade
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Checked data types.
* Checked for missing values.
* Checked for duplicate records.
* Converted categorical columns to categorical data types during data handling.
* Applied **One-Hot Encoding** to categorical variables.
* Kept numerical variables in numerical form.
* Excluded `G1` and `G2` from the prediction features.

The dataset had **no missing values and no duplicate records**, so no imputation was required.

After encoding:

**30 original input features → 56 encoded features**

---

## 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and relationships between variables.

Visualizations include:

* Final Grade (`G3`) distribution
* Study Time vs Final Grade
* Previous Failures vs Final Grade
* Absences vs Final Grade
* Numerical feature correlation heatmap

Libraries used:

* Matplotlib
* Seaborn
* Pandas

---

## 🤖 Machine Learning Models

### 1. Linear Regression

Linear Regression was used as the **baseline model** because the target variable, G3, is continuous.

### 2. Random Forest Regression

Random Forest Regression was implemented because student performance may involve non-linear relationships between different factors.

The model was configured with:

* **200 decision trees**
* `random_state = 42`

---

## 📊 Model Evaluation

The models were evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score**

| Model                    |  MAE | RMSE |   R² |
| ------------------------ | ---: | ---: | ---: |
| Linear Regression        | 3.40 | 4.20 | 0.14 |
| Random Forest Regression | 2.97 | 3.75 | 0.31 |

Random Forest showed lower MAE and RMSE and a higher R² score on the test set, so it was used in the final application.

> **Note:** R² = 0.31 means the model explains approximately 31% of the variation in the test-set target values. It should not be interpreted as 31% prediction accuracy.

---

## 💾 Model Persistence

The trained model and preprocessing object were saved using **Joblib**.

```text
random_forest_model.pkl
preprocessor.pkl
```

This allows the Streamlit application to load the trained model directly without retraining it every time.

The preprocessor is also saved so that new user inputs receive the same preprocessing used during model development.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit web application.

### Application Flow

```text
User enters student details
          ↓
Pandas DataFrame
          ↓
Saved Preprocessor
          ↓
Random Forest Model
          ↓
Predicted Final Grade (G3)
```

The application accepts student-related information such as:

* Age
* Gender
* Study time
* Previous failures
* Family support
* Internet access
* Absences
* Family relationship quality
* And other student-related factors

The application then displays the predicted final grade on a scale of 0–20.

---

## 💡 Potential Real-World Use

The system can be used as an **early performance prediction tool** in educational environments.

For example, a school or college could use similar systems to identify students who may require additional academic support and take appropriate preventive measures.

This project is a prototype and would require further validation and deployment testing before being used for real educational decisions.

---

## 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming               |
| Pandas       | Data manipulation         |
| Scikit-learn | Machine Learning          |
| Matplotlib   | Data visualization        |
| Seaborn      | Exploratory Data Analysis |
| Streamlit    | Web application           |
| Joblib       | Model persistence         |

---

## 📁 Project Structure

```text
StudentPerformance/
│
├── student_data.csv
├── data_analysis.py
├── data_cleaning.py
├── eda.py
├── preprocessing.py
├── model.py
├── app.py
│
├── random_forest_model.pkl
├── preprocessor.pkl
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder

```bash
cd StudentPerformance
```

### 3. Install the required libraries

```bash
pip install pandas scikit-learn matplotlib seaborn streamlit joblib
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 🔮 Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Cross-validation
* Trying additional regression algorithms
* Improving model performance
* Adding prediction confidence or uncertainty information
* Creating a more detailed student performance dashboard
* Deploying the Streamlit application online
* Using a larger and more diverse dataset

---

## 👨‍💻 Author

**Anu G**

B.Tech — Computer Science and Engineering (Data Science)

---

## ⭐ Project Highlights

* End-to-end Machine Learning project
* Data preprocessing and exploratory analysis
* Categorical feature encoding
* Model comparison
* Random Forest Regression
* Model persistence using Joblib
* Interactive Streamlit deployment
