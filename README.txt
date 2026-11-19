Machine Learning Web Application (TP1 Perceptron + TP2 Linear Regression)

Install Python (3.10+ recommended), then from this folder:

  python -m pip install -r requirements.txt
  python -m streamlit run app.py

Streamlit opens the site in your browser (normally http://localhost:8501).

TP1:
- Binary perceptron: Setosa/Versicolor, petal length and petal width.
- Multiclass one-vs-rest perceptron: all 3 Iris species, standardized 4 features.
- Shows a visual flower reference for the predicted class.

TP2:
- Diabetes simple regression (BMI feature; dataset values standardized).
- Diabetes multiple regression (10 standardized features).
- Iris simple regression (petal length -> sepal length).
- Iris multiple regression (sepal width, petal length, petal width -> sepal length).

Models are trained on the sklearn datasets during app initialization and predictions use the trained models.
The Diabetes regression output is a dataset target score and not a medical diagnosis.
