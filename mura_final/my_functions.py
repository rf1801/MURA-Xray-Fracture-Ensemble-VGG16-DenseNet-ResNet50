from my_imports import *
mura_location = "C:/Users/raouf/Desktop/pfe/"
def load_data(file, part):
    train_data = pd.read_csv(file)
    train_data = train_data.rename(columns={train_data.columns[0]: 'image_path'})
    train_data['label'] = 2
    string_to_add = mura_location
    train_data['image_path'] = string_to_add + train_data['image_path'].astype(str)
    for index, row in train_data.iterrows():
        if 'positive' in str(row['image_path']).lower():
            train_data.at[index, 'label'] = "positive"
        else:
            train_data.at[index, 'label'] = "negative"

    if part == 0:
        pass
    else:
        train_data = train_data[train_data['image_path'].str.contains(part)]
    return train_data

class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name="f1", initializer="zeros")
        self.precision_fn = tf.keras.metrics.Precision()
        self.recall_fn = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        self.f1.assign(2 * ((p * r) / (p + r + 1e-10)))

    def result(self):
        return self.f1

    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)


def save_history(name,history_df):

    history_excel_path = (
            save_path +
            name +
            '_history' 
            '.xlsx'
    )

    history_df.to_excel(history_excel_path, index=False)
    print("Training history saved to:", history_excel_path)






def plot_metrics(history: tf.keras.callbacks.History,
                 metrics: list = ["loss", "cohen_kappa", "precision", "recall"],
                 ) -> None:
    plt.rcParams["figure.figsize"] = (18, 15)

    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(
            history.epoch, history.history[metric], linewidth=1.8, label="training"
        )
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            linestyle="--",
            linewidth=1.8,
            label="validation",
        )
        plt.xlabel("epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "cohen_kappa":
            plt.ylim([-1, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()



class CustomCallback(keras.callbacks.Callback):
    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print(f"tp          :", logs["tp"])
        print(f"fp          :", logs["fp"])
        print(f"tn          :", logs["tn"])
        print(f"fn          :", logs["fn"])
        print(f"binary_acc  :", logs["binary_acc"])
        print(f"precision   :", logs["precision"])
        print(f"recall      :", logs["recall"])
        print(f"f1_score    :", logs["f1_score"])
        print(f"roc_auc     :", logs["roc_auc"])
        print(f"pr_auc      :", logs["pr_auc"])
        print(f"cohen_kappa :", logs["cohen_kappa"])

def build_ensemble_model():
    inputs = tf.keras.Input(shape=input_shape)

    resnet_preprocess_input = preprocess_input_resnet(inputs)
    resnet_base_output = resnet_base(resnet_preprocess_input)

    densenet_preprocess_input = preprocess_input_densenet(inputs)
    densenet_base_output = densenet_base(densenet_preprocess_input)

    concat_output = tf.keras.layers.Concatenate(axis=-1)([resnet_base_output, densenet_base_output])

    x = Dense(2048, activation="relu")(concat_output)
    x = Dropout(0.8)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, prediction)
    model._name = "Ensemble_model"

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=METRICS)
    return model