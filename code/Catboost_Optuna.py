import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold


def run():
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    column_names = [f'feature_{x}' for x in range(75)]
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    submission_sample = pd.read_csv('./sample_submission.csv')
    test = test.drop(['id'], axis=1)
    X = train.drop(['target', 'id'], axis=1)
    Y = train['target']
    Y = label_encoder.fit_transform(Y)
    X[column_names] = scaler.fit_transform(X[column_names])
    test[column_names] = scaler.transform(test[column_names])


    def objective(trial, data=X, target=Y):
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
        params = {'iterations': trial.suggest_int("iterations", 4000, 15000),
                  'od_wait': trial.suggest_int('od_wait', 500, 2300),
                  'loss_function': 'MultiClass',
                  'task_type': "GPU",
                  'eval_metric': 'MultiClass',
                  'leaf_estimation_method': 'Newton',
                  'bootstrap_type': 'Bernoulli',
                  'learning_rate': trial.suggest_uniform('learning_rate', 0.02, 0.3),
                  'reg_lambda': trial.suggest_uniform('reg_lambda', 1e-5, 100),
                  'subsample': trial.suggest_uniform('subsample', 0, 1),
                  'random_strength': trial.suggest_uniform('random_strength', 10, 30),
                  'depth': trial.suggest_int('depth', 1, 6),
                  'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
                  'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 5, 10),
                  'num_leaves': trial.suggest_int('num_leaves', 50, 64),
                  'grow_policy': 'Lossguide'
                  }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
        y_preds = model.predict_proba(X_test)
        log_loss_multi = log_loss(y_test, y_preds)
        return log_loss_multi


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print(study.best_trial)
    cat_params = study.best_trial.params
    cat_params['loss_function'] = 'MultiClass'
    cat_params['eval_metric'] = 'MultiClass'
    cat_params['bootstrap_type'] = 'Bernoulli'
    cat_params['leaf_estimation_method'] = 'Newton'
    cat_params['random_state'] = 42
    cat_params['task_type'] = 'GPU'
    cat_params['grow_policy'] = 'Lossguide'
    test_preds = None

    # 十次训练 求取均值
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    print("Catoost____Start____Prediction")
    for fold, (tr_index, val_index) in enumerate(kf.split(X.values, Y)):
        print(f"Catoost____Fold____{fold}")
        x_train, x_val = X.values[tr_index], X.values[val_index]
        y_train, y_val = Y[tr_index], Y[val_index]
        eval_set = [(x_val, y_val)]
        model = CatBoostClassifier(**cat_params)
        model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
        train_preds = model.predict(x_train)
        val_preds = model.predict_proba(x_val)
        print(log_loss(y_val, val_preds))
        if test_preds is None:
            test_preds = model.predict_proba(test[column_names].values)
        else:
            test_preds += model.predict_proba(test[column_names].values)
    test_preds /= 10
    submission = pd.DataFrame(test_preds, columns=[f'Class_{x + 1}' for x in range(9)])
    submission.insert(0, 'id', submission_sample['id'])
    submission.to_csv('Catboost_Optune_test.csv', index=False)
    print(f"Catboost_Optune_test.csv has been created.")


if __name__ == '__main__':
    run()