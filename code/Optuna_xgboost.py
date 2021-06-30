import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


# 与对CatBoost 进行的调节类似
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
        params = {'booster': 'gbtree',
                  'nthread': 4,
                  'use_label_encoder': False,
                  'eta': trial.suggest_uniform('eta', 1e-5, 0.3),
                  'gamma': trial.suggest_uniform('gamma', 0.1, 0.6),
                  'max_depth': trial.suggest_int('max_depth', 3, 8),
                  'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
                  # 'sub_sample': trial.suggest_uniform('sub_sample', 0.4, 0.9), not used
                  'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.2, 0.8),
                  'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.2, 0.8),
                  'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.2, 0.8),
                  'reg_lambda': trial.suggest_uniform('reg_lambda', 20, 80),
                  'reg_alpha': trial.suggest_uniform('reg_alpha', 0.001, 10),
                  'objective': 'multi:softmax',
                  'eval_metric': 'mlogloss'
                  }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
        y_preds = model.predict_proba(X_test)
        log_loss_multi = log_loss(y_test, y_preds)
        return log_loss_multi


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print(study.best_trial)
    xgb_params = study.best_trial.params
    xgb_params['booster'] = 'gbtree'
    xgb_params['objective'] = 'multi:softmax'
    xgb_params['nthread'] = 4
    xgb_params['use_label_encoder'] = False
    xgb_params['eval_metric'] = 'mlogloss'
    test_preds = None
    # 十次训练 求取均值
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    print("Xgboost____Start____Prediction")
    for fold, (tr_index, val_index) in enumerate(kf.split(X.values, Y)):
        print(f"Xgboost____Fold____{fold}")
        x_train, x_val = X.values[tr_index], X.values[val_index]
        y_train, y_val = Y[tr_index], Y[val_index]
        eval_set = [(x_val, y_val)]
        model = xgb.XGBClassifier(**xgb_params)
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
    submission.to_csv('Xgboost_Optune_test.csv', index=False)
    print(f"Xgboost_Optune_test.csv has been created.")


if __name__ == '__main__':
    run()