import pandas as pd
import os

general_dataset_path = "/home/dfoundry/dev/projects/" \
                       "Heart_Disease_Classification/" \
                       "data_files/heart_disease_dataset.csv"

general_dataset_directory = "/home/dfoundry/dev/projects/" \
                            "Heart_Disease_Classification/data"


class DataWork:

    def add_new_data(self, data_for_adding, saving_name,
                     old_dataset_path=general_dataset_path,
                     saving_folder_path=general_dataset_directory):
        old_dataset_df = pd.read_csv(old_dataset_path)
        new_dataset = self.crate_data_for_prediction(data_for_adding)
        new_added_dataset = old_dataset_df.append(new_dataset, ignore_index=True)
        saving_path = os.path.join(saving_folder_path, (saving_name + ".csv"))
        new_added_dataset.to_csv(saving_path)
        return "Done"

    def crate_data_for_prediction(self, data_for_prediction: dict) -> pd.DataFrame:
        """
        From dictionary create DataFrame.
        :param data_for_prediction:
        :return:
        """
        df_for_prediction = pd.DataFrame(data_for_prediction, index=[0])
        return df_for_prediction


