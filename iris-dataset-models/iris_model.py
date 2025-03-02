from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class IrisModel:
    def __init__(self):
        # load iris dataset from sklearn.datasets
        self.iris = datasets.load_iris()
        X = self.iris.data
        y = self.iris.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)

        # Create a Gaussian Classifier
        self.clf = RandomForestClassifier()

        # Train the model using the training sets
        self.clf.fit(self.X_train, self.y_train)
        # Predict the response for test dataset
        y_pred = self.clf.predict(self.X_test)
        # Model Accuracy, how often is the classifier correct?
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred)}")

        print("Features:", self.iris.feature_names) # Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        print("Labels:", self.iris.target_names) # Labels: ['setosa' 'versicolor' 'virginica']
    
    def predict_species(self, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm):
        predicted_species = self.clf.predict([[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]])
        return self.iris.target_names[predicted_species][0]
        # return predicted_species
  
iris_model = IrisModel()
print(f"Predicted species: {iris_model.predict_species(6.3, 3.3, 6.0, 2.5)}")
# print(f"Predicted species: {iris_model.predict_species(5.4, 3.7, 1.5, 0.2)}")

# Save the trained model
# joblib.dump(clf, 'iris_model.pkl')
# print("Model saved!")