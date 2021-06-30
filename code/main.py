import Catboost_Optuna
import keras_network
import Decision_Forest
import Optuna_xgboost
import ensemble_realize
import ensemble_cat_NN

if __name__ == '__main__':
    Catboost_Optuna.run()
    keras_network.run()
    Decision_Forest.run()
    Optuna_xgboost.run()
    ensemble_cat_NN.run()
    ensemble_realize.run()
    print('Done!')
