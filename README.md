Iris Flower Classification System
This project is a machine learning-based classification system that predicts the species of Iris flowers using various classification algorithms. The project was developed as part of an internship at NOVANECTAR SERVICES PVT LTD.

Project Overview
The Iris dataset is a famous dataset used in machine learning for classification tasks. This system classifies flowers into three species:

Setosa
Versicolor
Virginica
The project uses features such as:

Sepal length
Sepal width
Petal length
Petal width
The system implements multiple machine learning algorithms for classification, including:

Logistic Regression
Decision Tree
K-Nearest Neighbors (KNN)
Key Features
User-friendly Input: Users can input flower measurements via the console, and the system will predict the flower species.
Multiple Classifiers: The system is trained using three different classifiers and selects the most accurate one.
Hyperparameter Optimization: The KNN classifier is tuned using GridSearchCV to achieve optimal performance.
Performance Metrics: The models are evaluated using accuracy, precision, recall, and F1 score.
How to Use
Clone the Repository:

bash
Copy code
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification
Install the Required Libraries: Make sure you have the necessary dependencies installed. You can install them using the following command:

bash
Copy code
pip install -r requirements.txt
Run the Classifier: To classify an Iris flower, run the iris_classification.py file:

bash
Copy code
python iris_classification.py
Input Measurements: Enter the sepal and petal measurements when prompted, and the system will return the predicted species of the flower.
