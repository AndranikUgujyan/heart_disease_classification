from flask import Flask, request, jsonify
import pandas as pd
from heart_disease.add_and_create_dataset import DataWork
from heart_disease.prediction import DataPredict
from heart_disease.data_preprocessing import DataPreprocessing
from heart_disease.train_and_hyperparameter_tuning import HyperTuning

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def data_predict():
    dw = DataWork()
    dp = DataPredict()
    try:
        data_come = dict(request.get_json())
        coming_data_df = dw.crate_data_for_prediction(data_come)
        output_result = dp.prediction(coming_data_df)
        return pd.Series(output_result).to_json(orient='values')
    except ValueError:
        return jsonify("Values can not be empty")


@app.route("/train", methods=['POST'])
def data_train():

    madel_name = str(dict(request.get_json())['model_name'])
    dpre = DataPreprocessing()
    data = dpre.train_and_test_data()
    x_train = data.x_train
    x_test = data.x_test
    y_train = data.y_train
    y_test = data.y_test
    hp = HyperTuning()
    result = hp.final_model(madel_name, x_train, y_train, x_test, y_test)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
