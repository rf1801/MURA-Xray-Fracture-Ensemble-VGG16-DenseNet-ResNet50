from my_imports import *

main_save_path = "C:/Users/raouf/Desktop/pfe/history/cheat/resnet/"
mura_location = "C:/Users/raouf/Desktop/pfe/"
data_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/'
train_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/train'
test_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid'
train_file = "C:/Users/raouf/Desktop/pfe/MURA-v1.1/train_image_paths.csv"
val_file = "C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid_image_paths.csv"


target_categories = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER',  'XR_WRIST']
batch_size = 16
num_epochs = 100
fine_tuning_epochs = 50
validation_split = 0.2
loss_function = tf.keras.losses.BinaryCrossentropy()
learning_rate = 0.0001
input_shape = (224, 224, 3)
architecture = 'ResNet'

include_top = False
early_stopping_patience = 15
shuffle = True
class_mode = 'binary'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
trainable = False
training = False
dropout = 0.2


def save_history(name, history_df):
    history_excel_path = save_path + name + '_history' + '.xlsx'
    history_df.to_excel(history_excel_path, index=False)
    print("Training history saved to:", history_excel_path)


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
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)


def plot_metrics(history: tf.keras.callbacks.History,
                 metrics: list = ["loss", "binary_acc", "precision", "recall"]) -> None:
    plt.rcParams["figure.figsize"] = (18, 15)
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], linewidth=1.8, label="training")
        plt.plot(history.epoch, history.history["val_" + metric], linestyle="--", linewidth=1.8, label="validation")
        plt.xlabel("epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0, 1])
        plt.legend()


def my_func(model, data_generator, threshold=0.5):
    y_pred_prob = model.predict(data_generator)
    y_pred = (y_pred_prob > threshold).astype(int)
    y_true = data_generator.classes
    print("Classification Report:")
    print(classification_report(y_true, y_pred))


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


METRICS = [
    tf.keras.metrics.TruePositives(name="tp"),
    tf.keras.metrics.FalsePositives(name="fp"),
    tf.keras.metrics.TrueNegatives(name="tn"),
    tf.keras.metrics.FalseNegatives(name="fn"),
    tf.keras.metrics.BinaryAccuracy(name="binary_acc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    F1Score(name="f1_score"),
    tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
    tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    tfa.metrics.CohenKappa(name="cohen_kappa", num_classes=2),
]

resnet = tf.keras.applications.resnet50.ResNet50(include_top=include_top, weights="imagenet",
                                                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), pooling="max", classes=2)


def make_resnet_model(base_model: tf.keras.applications.resnet50.ResNet50) -> tf.keras.Model:
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    x = base_model(inputs, training=training)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    output = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_function, metrics=METRICS)
    return model


new_model = make_resnet_model(base_model=resnet)
new_model.summary()

early_stopping = EarlyStopping(monitor="val_binary_acc", verbose=1, patience=early_stopping_patience, mode="max",
                               baseline=0.0, restore_best_weights=True)




class CustomCallback(keras.callbacks.Callback):
    def __init__(self, test_data, save_path, model_name):
        self.test_data = test_data
        self.save_path = save_path
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        global best  # Declare best_acc as a global variable
        test_loss, test_tp, test_fp, test_tn, test_fn, test_binary_acc, test_precision, test_recall, test_f1_score, test_roc_auc, test_pr_auc, test_cohen_kappa = self.model.evaluate(
            self.test_data, verbose=0)
        model_filename = f'{self.save_path}{self.model_name}_epoch_{epoch + 1}_test_acc_{test_binary_acc:.4f}_kappa_{test_cohen_kappa:.4f}.keras'

        metric_to_use = test_binary_acc + test_cohen_kappa
        if metric_to_use > best:
            best = metric_to_use  # Update best_acc if the current test accuracy is higher
            self.model.save(model_filename)
            print(f'\nTest accuracy after epoch {epoch + 1}: {test_binary_acc:.4f} - Model saved as {model_filename}')
        else:
            print(f'\nTest accuracy after epoch {epoch + 1}: {test_binary_acc:.4f}  - model not saved')





for target_category in target_categories:
    best = 0
    save_path = main_save_path + target_category + "/"
    model_name = architecture + "_" + target_category
    #model_checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_binary_acc", save_best_only=True)

    df_train = load_data(train_file, target_category)
    df_test = load_data(val_file, target_category)
    df_train, df_valid = train_test_split(df_train, test_size=validation_split, random_state=42, shuffle=True)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30, horizontal_flip=True,
                                                              fill_mode="constant", cval=0.0,
                                                              preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
    train_generator = datagen.flow_from_dataframe(df_train, directory=None, x_col='image_path', y_col='label',
                                                  target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=batch_size,
                                                  class_mode=class_mode)
    validation_generator = datagen.flow_from_dataframe(df_valid, directory=None, x_col='image_path', y_col='label',
                                                       target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=batch_size,
                                                       class_mode=class_mode)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
    test_generator = test_datagen.flow_from_dataframe(df_test, directory=None, x_col='image_path', y_col='label',
                                                      target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=batch_size,
                                                      class_mode=class_mode)

    custom_callback = CustomCallback(test_generator, save_path, model_name)

    history = new_model.fit(train_generator, validation_data=validation_generator, epochs=num_epochs, shuffle=shuffle,
                            callbacks=[early_stopping, custom_callback])

    #plot_metrics(history)


    #new_model.evaluate(test_generator)



    y_pred_prob = new_model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = test_generator.classes

    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    resnet.trainable = True

    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            1e-5
        ),  # we need small learning rate to avoid catastrophic forgetting
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS,
    )

    model_name = 'finetuned_' + architecture + "_" + target_category

    model_path = save_path + '_best_' + model_name + '.keras'
    model_checkpoint = ModelCheckpoint(filepath=model_path,
                                       monitor="val_cohen_kappa",
                                       save_best_only=True)

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_cohen_kappa",
        verbose=1,
        patience=4,
        mode="max",
        factor=0.2,
        min_lr=1e-10,
    )

    custom_callback = CustomCallback(test_generator, save_path, model_name)
    history = new_model.fit(train_generator,
                            # steps_per_epoch=len(df_train) // batch_size,
                            validation_data=validation_generator,
                            # validation_steps=len(df_valid) // batch_size,
                            epochs=fine_tuning_epochs,
                            shuffle=shuffle,
                            callbacks=[early_stopping, reduce_lr_on_plateau,custom_callback])

    #new_model.save(save_path + model_name + '.keras')
    #plot_metrics(history)


    #new_model.evaluate(test_generator)



    # Make predictions using the test generator
    y_pred_prob = new_model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_test = test_generator.classes
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

