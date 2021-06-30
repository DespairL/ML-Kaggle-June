import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks


def Build_Model():
    model = keras.Sequential([
        layers.Input(shape=(75,)),
        layers.Embedding(400, 8, input_length=256),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(9, activation='softmax'),
    ])
    return model


def prediction(X_train, Y_train, X_test, callbacks):
    keras.backend.clear_session()  # 进行keras的交叉验证时,使每一次训练独立
    kfold = StratifiedKFold(n_splits=10)
    y_pred = np.zeros((100000, 9))
    train_oof = np.zeros((200000, 9))
    count = 0
    for (train_idx, val_idx) in kfold.split(X_train, Y_train):
        print(f"Keras____Network____Fold____{count}")
        x_train = X_train.iloc[train_idx]
        y_train = Y_train.iloc[train_idx]
        x_val = X_train.iloc[val_idx]
        y_val = Y_train.iloc[val_idx]
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        model = Build_Model()
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002), metrics='accuracy')
        model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=False, validation_data=(x_val, y_val), callbacks=callbacks)
        y_pred += model.predict(X_test) / kfold.n_splits
        val_pred = model.predict(x_val)
        train_oof[val_idx] = val_pred
        log_loss = metrics.log_loss(y_val, val_pred)
        print(f"Logloss: {log_loss:0.5f}")
        count += 1
    return y_pred, train_oof


def run():
    df_train = pd.read_csv('./train.csv', index_col='id')
    Y_train = df_train['target']
    X_train = df_train.drop(columns='target')
    X_test = pd.read_csv('./test.csv', index_col='id')
    Columns = [f'Class_{x}' for x in range(1, 10)]
    class_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2,
                 'Class_4': 3, 'Class_5': 4, 'Class_6': 5,
                 'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
    Y_train = Y_train.map(class_map).astype('int')
    Y_train = to_categorical(Y_train)
    keras.backend.clear_session()
    model = Build_Model()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                  metrics='accuracy')
    early_stopping = callbacks.EarlyStopping(patience=20, min_delta=0.0000001, restore_best_weights=True, )
    plateau = callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_delt=0.0000001, cooldown=0, verbose=False)
    Y_train = df_train['target'].copy()
    Y_train = Y_train.map(class_map).astype('int')
    print("Keras____Network____Start____Prediction")
    Final_prediction, train_oof = prediction(X_train, Y_train, X_test, callbacks=[early_stopping, plateau])
    print(f"Final Logloss: {metrics.log_loss(Y_train, train_oof):0.6f}")
    Output = pd.DataFrame(Final_prediction, columns=Columns)
    Output['id'] = X_test.index
    Output.to_csv('keras_network_test.csv', index=False)
    print(f"keras_network_test.csv has been created.")


if __name__ == '__main__':
    run()