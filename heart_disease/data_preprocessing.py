from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

Data = namedtuple('Dataset', 'x_train, x_test, y_train, y_test')

General_data_path = "/home/dfoundry/dev/projects/" \
                    "Heart_Disease_Classification/" \
                    "data_files/heart_disease_dataset.csv"


class DataPreprocessing:

    def train_and_test_data(self, dataset_of_heart_disease_path=General_data_path):
        """
        Read dataset from path in DataFrame and
        return x_train, x_test, y_train, y_test in namedtuple.

        :param dataset_of_heart_disease_path:
        :return:
        """
        independent_data_x, target_data_y = self.split_independent_and_target(dataset_of_heart_disease_path)
        train_test_data = self.create_train_and_test_data(independent_data_x, target_data_y)
        return train_test_data

    def split_independent_and_target(self, path_of_heart_disease_dataset):
        """
        From path of dataset create DataFrame.
        Split Dataset to independent and target DataFrames.
        :param path_of_heart_disease_dataset:
        :return:
        """
        heart_disease_df = pd.read_csv(path_of_heart_disease_dataset)
        if 'target' in list(heart_disease_df):
            independent_variables_x = heart_disease_df.drop("target", axis=1)
            target_variable_y = heart_disease_df["target"]
            return independent_variables_x, target_variable_y

    def create_train_and_test_data(self,
                                   independent_data: pd.DataFrame,
                                   target_data: pd.DataFrame,
                                   size_of_test=0.2) -> namedtuple:
        """
        From independent and target DataFrames create independent_train (x_train),
        independent_test (x_test), target_train (y_train), target_test (y_test)
        and return in factory function namedtuple.
        :param independent_data:
        :param target_data:
        :param size_of_test:
        :return:
        """

        independent_train, independent_test, target_train, target_test = \
            train_test_split(independent_data, target_data, test_size=size_of_test)
        x_train = independent_train
        x_test = independent_test
        y_train = target_train
        y_test = target_test
        return Data(x_train, x_test, y_train, y_test)



