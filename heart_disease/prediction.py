from joblib import load
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from heart_disease.data_preprocessing import DataPreprocessing

data = DataPreprocessing().train_and_test_data()
x_test = data.x_test
y_test = data.y_test

basic_model_path = "/home/dfoundry/dev/projects/" \
                   "Heart_Disease_Classification/" \
                   "models/main_trained_model_model.joblib"


class DataPredict:

    def prediction(self, data_for_prediction,
                   chosen_model=basic_model_path) -> np.ndarray:
        """
        Predict target base on trained model.
        (Defold use basic trained model)
        :param data_for_prediction:
        :param chosen_model:
        :return:
        """
        basic_joblib_model = load(filename=chosen_model)
        predicted_target = basic_joblib_model.predict(data_for_prediction)
        return predicted_target

    def conf_matrix(self, test_data_x=x_test, test_data_y=y_test):
        """
        Plots confusion matrix using Seaborn.
        :param test_data_x:
        :param test_data_y:
        :return:
        """
        predicted_y = self.prediction(test_data_x)
        plt.subplots(figsize=(3, 3))
        plt.subplots(figsize=(3, 3))

        ax = sns.heatmap(confusion_matrix(test_data_y, predicted_y),
                         annot=True,
                         cbar=False)
        plt.xlabel("True label")
        plt.ylabel("Predicted label")

        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()

    def classifi_report(self, test_data_x_cl=x_test, test_data_y_cl=y_test):
        """
        Return classification report base on test data.
        :param test_data_x_cl:
        :param test_data_y_cl:
        :return:
        """
        pred_y = self.prediction(test_data_x_cl)
        classification_report_result = classification_report(pred_y, test_data_y_cl)
        return classification_report_result



