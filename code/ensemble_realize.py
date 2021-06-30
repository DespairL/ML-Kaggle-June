import pandas as pd
import numpy as np


def run():
    # Catboost and Keras_Network have been ensembled in ensemble_cat_NN.py
    Catboost_result = pd.read_csv('./Catboost_Optune_test.csv')
    Xgboost_result = pd.read_csv('./Xgboost_Optune_test.csv')
    # Lightgbm_result = pd.read_csv('./lightgbm_Optune_v1.csv')
    # Keras_Network_result = pd.read_csv('./keras_network_test.csv')
    Decision_Tree_result = pd.read_csv('./Decision_Forest_test.csv')
    Ensemble_NN_Cat_result = pd.read_csv('./ensemble_data_nn_cat.csv')

    # Catboost_pred = Catboost_result.drop(columns='id')
    Xgboost_pred = Xgboost_result.drop(columns='id')
    # Lightgbm_pred = Lightgbm_result.drop(columns='id')
    # Keras_Network_pred = Keras_Network_result.drop(columns='id')
    Decision_Tree_pred = Decision_Tree_result.drop(columns='id')
    Ensemble_NN_Cat_pred = Ensemble_NN_Cat_result.drop(columns='id')

    # Lower weights of worse model
    Combine_result = 0.1 * Xgboost_pred + 0.1 * Decision_Tree_pred + 0.8 * Ensemble_NN_Cat_pred
    # use clip
    # Combine_result = np.clip(Combine_result, 0.05, 0.95)
    # clip seems no use for this task
    Columns = [f'Class_{x}' for x in range(1, 10)]
    Out_DataFrame = pd.DataFrame(Combine_result, columns=Columns)
    Out_DataFrame['id'] = Catboost_result['id']
    Out_DataFrame = Out_DataFrame[
        ['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']]
    Out_DataFrame.to_csv("final.csv", index=False)
    print(f"Ensemble Done! The Final csv : final.csv")

if __name__ == '__main__' :
    run()