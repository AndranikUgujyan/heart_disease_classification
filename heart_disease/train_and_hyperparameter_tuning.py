from collections import namedtuple
from heart_disease.data_preprocessing import DataPreprocessing
import numpy as np
import pandas as pd
import warnings
import os
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
warnings.filterwarnings("ignore")

model_directory_path = "/home/dfoundry/dev/projects/" \
                       "Heart_Disease_Classification/models"

training_models = {"Logistic Regression": LogisticRegression(),
                   "KNN": KNeighborsClassifier(),
                   "Random Forest": RandomForestClassifier()}

# Create a hyperparameter grid for LogisticRegression
log_reg_grid_param = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid_param = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}

model = namedtuple('Model', 'parameters, score, fitted_model')


class HyperTuning:

    def final_model(self, new_model_name: str,
                    indep_train: pd.DataFrame,
                    targ_train: pd.DataFrame,
                    indep_test: pd.DataFrame,
                    targ_test: pd.DataFrame) -> str:
        """
        Compare accuracy scores of hyperparameters tuned and without tuned models
        and save it as a joblib file by given name.
        :param new_model_name:
        :param indep_train:
        :param targ_train:
        :param indep_test:
        :param targ_test:
        :return:
        """

        model_fir = {}
        basic_model_scores = {}
        for name_of_model, one_model in training_models.items():
            basic_fitted_model = training_models[name_of_model].fit(indep_train, targ_train)
            model_fir[name_of_model] = basic_fitted_model
            basic_model_scores[name_of_model] = basic_fitted_model.score(indep_test, targ_test)

        hyper_tuned_model = self.log_hyper_tun(indep_train, targ_train, indep_test, targ_test)

        if max(basic_model_scores, key=basic_model_scores.get) == "Logistic Regression" \
                and basic_model_scores["Logistic Regression"] <= hyper_tuned_model.score:
            new_model_filename = "{}_model.joblib".format(new_model_name)
            new_model_path = os.path.join(model_directory_path, new_model_filename)
            dump(hyper_tuned_model.fitted_model, filename=new_model_path)
            return "Hyper-tuned model trained and saved"
        else:
            basic_model = model_fir["Logistic Regression"]
            basic_model_filename = "{}_model.joblib".format("basic")
            new_model_path = os.path.join(model_directory_path, basic_model_filename)
            dump(basic_model, filename=new_model_path)
            return "Basic model trained and saved"

    def log_hyper_tun(self, independent_train: pd.DataFrame,
                      target_train: pd.DataFrame,
                      independent_test: pd.DataFrame,
                      target_test: pd.DataFrame) -> namedtuple:
        """
        Use hyperparameter grids for random hyperparameter
        search for LogisticRegression and RandomForestClassifier
        and chose the model with the best hyperparameters.
        :param independent_train:
        :param target_train:
        :param independent_test:
        :param target_test:
        :return:
        """
        rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                        param_distributions=log_reg_grid_param,
                                        cv=5,
                                        n_iter=20,
                                        verbose=True)
        rs_fitted_model = rs_log_reg.fit(independent_train, target_train)
        rs_parameters = rs_fitted_model.best_params_
        rs_score = rs_log_reg.score(independent_test, target_test)

        gs_log_reg = GridSearchCV(LogisticRegression(),
                                  param_grid=log_reg_grid_param,
                                  cv=5,
                                  verbose=True)

        gs_fitted_model = gs_log_reg.fit(independent_train, target_train)
        gs_parameters = gs_fitted_model.best_params_
        gs_score = gs_log_reg.score(independent_test, target_test)
        if rs_score <= gs_score:
            gs_tuning_result = model(gs_parameters, gs_score, gs_fitted_model)
            return gs_tuning_result
        else:
            rs_tuning_result = model(rs_parameters, rs_score, rs_fitted_model)
            return rs_tuning_result



