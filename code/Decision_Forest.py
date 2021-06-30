import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# Two classes copy from https://keras.io/examples/structured_data/deep_neural_decision_forests/
class NeuralDecisionTree(keras.Model):
    def __init__(self, depth, num_features, used_features_rate, num_classes):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_classes = num_classes

        # Create a mask for the randomly selected features.
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indicies = np.random.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]

        # Initialize the weights of the classes in leaves.
        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=[self.num_leaves, self.num_classes]
            ),
            dtype="float32",
            trainable=True,
        )
        # Initialize the stochastic routing layer.
        self.decision_fn = layers.Dense(
            units=self.num_leaves, activation="sigmoid", name="decision"
        )

    def call(self, features):
        batch_size = tf.shape(features)[0]

        # Apply the feature mask to the input features.
        features = tf.matmul(
            features, self.used_features_mask, transpose_b=True
        )  # [batch_size, num_used_features]
        # Compute the routing probabilities.
        decisions = tf.expand_dims(
            self.decision_fn(features), axis=2
        )  # [batch_size, num_leaves, 1]
        # Concatenate the routing probabilities with their complements.
        decisions = layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )  # [batch_size, num_leaves, 2]
        mu = tf.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order.
        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = tf.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[
                              :, begin_idx:end_idx, :
                              ]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
        probabilities = keras.activations.softmax(self.pi)  # [num_leaves, num_classes]
        outputs = tf.matmul(mu, probabilities)  # [batch_size, num_classes]
        return outputs

class NeuralDecisionForest(keras.Model):
    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        super(NeuralDecisionForest, self).__init__()
        self.ensemble = []
        # Initialize the ensemble by adding NeuralDecisionTree instances.
        # Each tree will have its own randomly selected input features to use.
        for _ in range(num_trees):
            self.ensemble.append(
                NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
            )

    def call(self, inputs):
        # Initialize the outputs: a [batch_size, num_classes] matrix of zeros.
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, 9]) # * Change num_classes to 9 *

        # Aggregate the outputs of trees in the ensemble.
        for tree in self.ensemble:
            outputs += tree(inputs)
        # Divide the outputs by the ensemble size to get the average.
        outputs /= len(self.ensemble)
        return outputs


def run():
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    target = pd.get_dummies(train['target']) # 直接进行One-Hot编码
    forest_model = NeuralDecisionForest(num_trees=20, depth=5, num_features=20, used_features_rate=0.5, num_classes=9)
    metrics = [tf.keras.metrics.CategoricalCrossentropy()]
    loss = tf.keras.losses.CategoricalCrossentropy()
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.0000001, patience=2, verbose=0,
        mode='min', baseline=None, restore_best_weights=True)
    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, verbose=0,
        mode='min', min_delta=0.0000001, cooldown=0, min_lr=10e-7)
    N_FOLDS = 10
    SEED = 2021
    pred_forest = np.zeros((test.shape[0], 9))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    print("Decision____Tree____Start____Prediction")
    for fold, (tr_idx, ts_idx) in enumerate(skf.split(train, train.iloc[:, -1])):
        print(f"Decision____Tree____Fold____{fold}")
        X_train = train.iloc[:, 1:-1].iloc[tr_idx]
        y_train = target.iloc[tr_idx]
        X_test = train.iloc[:, 1:-1].iloc[ts_idx]
        y_test = target.iloc[ts_idx]
        inp = layers.Input(shape=(75,))
        x = layers.Embedding(400, 8, input_length=256)(inp)
        x = layers.Flatten()(x)
        API = layers.Dense(units=20, activation='relu', kernel_initializer='random_uniform',
                           bias_initializer=initializers.Constant(0.1))(x)
        model_forest = Model(inp, forest_model(API))
        model_forest.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss, metrics=metrics)
        model_forest.fit(X_train, y_train,
                         validation_data=(X_test, y_test),
                         batch_size=256,
                         epochs=50,
                         verbose=0,
                         callbacks=[es, plateau])
        pred_forest += model_forest.predict(test.iloc[:, 1:]) / N_FOLDS
        print(f"Decision____Tree____Fold____{fold}____Done!")
    Columns = [f'Class_{x}' for x in range(1, 10)]
    df_combined = pd.DataFrame(pred_forest, columns=Columns)
    df_combined['id'] = test['id']
    df_combined = df_combined[['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']]
    final_output = df_combined.to_csv('Decision_Forest_test.csv', index=False)
    print(f"Decision_Forest_test.csv has been created.")


if __name__ == '__main__':
    run()