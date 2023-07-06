import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
import helper_functions

def machine_learning_model(type):
    # Sets the models and their hyperparameters.
    svc = svm.SVC(random_state=1, class_weight="balanced", C=1, gamma=0.1)
    norm = Normalizer()
    pca = PCA(n_components=400, random_state=1)

    # Retrieves the data.
    print("Loading " + type + " Data...")
    x_train = pd.read_csv(filepath_or_buffer='./' + type + '/X_train.csv', names=helper_functions.COL, header=None)
    y_train = pd.read_csv(filepath_or_buffer='./' + type + '/Y_train.csv', names=["Class"], header=None)
    df_train = pd.concat([x_train, y_train], axis=1)

    # Cleans and splits the data.
    print("Cleaning " + type + " Data...")
    df_train = helper_functions.cleanData(df_train)
    y_train = df_train.pop("Class")
    x_train = df_train
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, stratify=y_train, random_state=1)

    # Performs feature elimination.
    print("Eliminating " + type + " Features...")
    x_train = helper_functions.featureElimination(x_train, True, pca, norm)
    x_test = helper_functions.featureElimination(x_test, False, pca, norm)

    # Trains the model.
    print("Training " + type + " Model...")
    svc.fit(x_train, y_train.values.ravel())
    print("Evaluating " + type + " Model...")

    # Uncomment out lines 39 - 43 to produce testing data.
    #  
    # x_test = pd.read_csv('./' + type + '/X_test.csv', names=helper_functions.COL, header=None)
    # x_test = helper_functions.cleanData(x_test)
    # x_test = helper_functions.featureElimination(x_test, False, pca, norm)
    # y_prob = svc.predict(x_test)
    # helper_functions.buildCSV(type, y_prob)

    # Evaluates the model.
    helper_functions.evaluation(svc, x_test, y_test)

if __name__ == "__main__":
    classes = ["multi", 'binary']
    for cla in classes:
        machine_learning_model(cla)