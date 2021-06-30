import pandas as pd
import numpy as np

def run():
    Catboost_result = pd.read_csv('./Catboost_Optune_test.csv')
    Keras_Network_result = pd.read_csv('./keras_network_test.csv')
    Catboost_pred = Catboost_result.drop(columns='id')
    Keras_Network_pred = Keras_Network_result.drop(columns='id')
    Final_ensemble_pred = 0.85 * Keras_Network_pred + 0.15 * Catboost_pred
    Columns = [f'Class_{x}' for x in range(1, 10)]
    Out_DataFrame = pd.DataFrame(Final_ensemble_pred, columns=Columns)
    Out_DataFrame['id'] = Catboost_result['id']
    Out_DataFrame = Out_DataFrame[
        ['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']]
    Out_DataFrame.to_csv("ensemble_data_nn_cat.csv", index=False)
    print(f"Ensemble Done! The Final csv : ensemble_data_nn_cat.csv")


if __name__ == '__main__':
    run()