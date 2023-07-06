import pandas as pd
import os
from helper_functions import evaluateModel
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures  # For advanced tasks.


# Global Variables
RANDOM_SEED = 1 # Seed ensures reproducibility with the results.
RATIO = 0.8 # Use 80% of the input data for training, 10% for validation and 10% for testing.
NUMBER_OF_FEATURES = 11 # Number of feature to be taken from feature selection.

def machine_learning_model():

    # 1: Data Processing
    print('Part One: Data Processing')
    # Load the data from the CSV file.
    dirname = os.path.dirname(__file__)
    df = pd.read_csv(dirname + '/drybeans.csv')

    # Dropping null data
    df = df.dropna()
    # Dropping duplicate rows
    df = df.drop_duplicates()

    # Splitting the data into input (x) and output (y)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Pipeline used for streamlining feature selection and scalers for each model.
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=chi2, k=NUMBER_OF_FEATURES)),
        ('scaler', StandardScaler()) 
        ])

    # Splitting the dataset into 3 sets: training, validation and testing based on the global variable RATIO. Currently 8:1:1 split.
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=RATIO, stratify=y, random_state=RANDOM_SEED)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, train_size=0.5, stratify=y_test, random_state=RANDOM_SEED)

    # For each model, independently apply Feature Selection and scale via Standardisation.
    x_train = pipeline.fit_transform(x_train, y_train)
    x_valid = pipeline.fit_transform(x_valid, y_valid)
    x_test = pipeline.fit_transform(x_test, y_test)

    # Part Two: Training
    print("Part Two: Training")
    # Used in 2a
    model_NoPenNoWeight = LogisticRegression(penalty='none', class_weight=None, max_iter=9000)
    # Used in 2a
    dummy_classifier = DummyClassifier(strategy='uniform')
    # Used in 2c
    model_NoPenBalanced = LogisticRegression(penalty='none', class_weight='balanced', max_iter=9000)
    # Used in 4a
    model_L2PenNoClassWeight = LogisticRegression(penalty='l2', class_weight=None, max_iter=9000)
    # Used in 4b
    # Polynomial Used for Advanced Tasks
    polynomial = PolynomialFeatures(degree=2, interaction_only=True)

    x_train_poly = pd.DataFrame(polynomial.fit_transform(x_train))
    x_valid_poly = pd.DataFrame(polynomial.fit_transform(x_valid))
    x_test_poly = pd.DataFrame(polynomial.fit_transform(x_test))

    print("Number of Features Before 2nd Degree Polynomial expansion: " + str(len(x_train[0])))
    print("Number of Features Resulting from a 2nd Degree Polynomial Expansion: " + str(len(x_train_poly.columns)) + "\n")
    model_L2PenBalanced = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=9000)
    model_L2PenBalanced_Poly = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=9000)

    # Fits each model to the data.
    model_NoPenNoWeight.fit(x_train, y_train)
    model_NoPenBalanced.fit(x_train, y_train)
    dummy_classifier.fit(x_train, y_train)
    model_L2PenNoClassWeight.fit(x_train, y_train)
    model_L2PenBalanced.fit(x_train, y_train)
    model_L2PenBalanced_Poly.fit(x_train_poly, y_train)

    # Part Three: Evaluation
    print("Part Three: Evaluation")

    evaluateModel(dummy_classifier, x_train, y_train, 'Training', 'Dummy', 'Dummy', False)

    evaluateModel(model_NoPenNoWeight, x_train, y_train, 'Training', 'none', 'none', False)
    evaluateModel(model_NoPenNoWeight, x_test, y_test, 'Testing', 'none', 'none', False)

    evaluateModel(model_NoPenBalanced, x_train, y_train, 'Training', 'none', 'Balanced', False)
    evaluateModel(model_NoPenBalanced, x_test, y_test, 'Testing', 'none', 'Balanced', False)

    evaluateModel(model_L2PenNoClassWeight, x_train, y_train, 'Training', 'L2', 'none', False)

    # For 4b
    evaluateModel(model_L2PenBalanced_Poly, x_train_poly, y_train, 'Training', 'L2', 'Balanced', False)
    
    # For 4c
    evaluateModel(model_L2PenBalanced, x_test, y_test, 'Testing', 'L2', 'Balanced', False)

    # For 4d
    evaluateModel(model_NoPenBalanced, x_test, y_test, 'Testing', 'none', 'Balanced', True)
    
if __name__ == "__main__":
    machine_learning_model()