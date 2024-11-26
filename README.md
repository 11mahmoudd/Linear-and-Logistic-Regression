# CO2 Emissions Prediction: Linear & Logistic Regression

This project focuses on predicting CO2 emissions and their respective classes for vehicles using **Linear Regression** and **Logistic Regression**. With climate change being a critical global issue, understanding the environmental impact of vehicles has become essential. This project uses machine learning techniques to analyze and predict the amount of CO2 emissions based on a dataset of over 7000 vehicle records.

## 🚗 **Dataset Overview**
The dataset, *co2_emissions_data.csv*, includes:
- **11 Features**: Vehicle make, model, size, engine size, cylinders, transmission type, fuel type, and fuel consumption ratings.
- **2 Target Variables**: 
  - **CO2 Emission Amount** (g/km): A continuous target for Linear Regression.
  - **Emission Class**: A categorical target for Logistic Regression.

---

## 🛠️ **Project Workflow**
### 1. **Data Loading & Exploration**
- Checked for missing values and ensured the data was clean.
- Assessed scaling across numeric features.
- Visualized:
  - **Pairplot** with histograms for diagonal subplots.
  - **Correlation Heatmap** to analyze relationships between numeric features.

### 2. **Data Preprocessing**
- Separated features and targets.
- Encoded categorical variables.
- Shuffled and split the data into **training** and **testing** sets.
- Scaled numeric features using training set statistics.

### 3. **Linear Regression**
- **Implemented from scratch** using **gradient descent** to predict the CO2 emission amount.
- Selected two strongly correlated but non-redundant features based on the correlation heatmap.
- Visualized the **cost function** improvement over iterations of gradient descent.
- Evaluated model performance using the **R² score**.

### 4. **Logistic Regression**
- Built a Logistic Regression model from scratch using **stochastic gradient descent** to classify emission classes.
- Used the same two features selected for Linear Regression.
- Evaluated the model using **accuracy** on the test set.

---

## 📊 **Key Highlights**
- **From Scratch Implementation**: Linear regression and logistic regression were implemented using Python’s core libraries (NumPy, Pandas, etc.).
- **Performance Evaluation**:
  - **Linear Regression**: R² score quantifying the model's predictive ability.
  - **Logistic Regression**: Accuracy score reflecting the classification model's correctness.
- **Visualization**: Insights from pairplots, heatmaps, and the gradient descent error curve.

---

## 🚀 **Results**
- The linear regression model successfully predicted CO2 emission amounts, showcasing a clear improvement in the cost function through gradient descent iterations.
- The logistic regression model classified vehicles’ emission classes with high accuracy.

---

## 💻 **Technologies Used**
- **Python**: Primary programming language.
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (for evaluation and preprocessing).
