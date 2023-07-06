import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report

CLASSES = ['BARBUNYA','BOMBAY','CALI','DERMASON','HOROZ','SEKER','SIRA']

def evaluateModel(model, x, y_true, set_type, penalty, class_weight, advanced):
    y_hat = model.predict(x)
    y_proba = model.predict_proba(x)

    model_string = set_type + ' Logistic Regression (Penalty = ' + penalty + ', Class Weight = ' + class_weight + ')'
    print(model_string)

    # Used for 3a
    accuracy = accuracy_score(y_true, y_hat)
    print('Classification Accuracy = {0:0.3f}'.format(accuracy))


    # Used for 3b
    balancedAccuracyScore = balanced_accuracy_score(y_true, y_hat)
    print('Balanced Accuracy = {0:0.3f}'.format(balancedAccuracyScore))


    # Used for 3c
    confusionMatrix = confusion_matrix(y_true, y_hat)
    print('Confusion Matrix = \n', confusionMatrix)

    # Used for 3d
    report = classification_report(y_true, y_hat, target_names=CLASSES)
    print('Classification Report = \n', report)

    if (advanced):
        y_test_bin = label_binarize(y_true, classes=CLASSES)
        n_classes = y_test_bin.shape[1]

        # Calculate precision and recall
        precision = dict()
        recall = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        for i in range(n_classes):
            plt.plot(recall[i], precision[i], lw=2, label=CLASSES[i].format(i))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc='lower left')
            plt.title('Precision-Recall curve')
            plt.show()