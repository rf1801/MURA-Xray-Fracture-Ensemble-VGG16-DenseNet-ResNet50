from my_imports2 import *
from my_functions2 import *


mura_location = "C:/Users/raouf/Desktop/pfe/"
data_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/'

train_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/train'
test_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid'



test_file = "C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid_image_paths.csv"


architecture = 'vgg'
main_save_path = "C:/Users/raouf/Desktop/pfe/history/new_arch/"+architecture+"/"


target_categories = ["XR_FINGER","XR_SHOULDER","XR_WRIST"]

batch_size = 16
num_epochs = 100
fine_tuning_epochs=50
validation_split = 0.2
loss_function = tf.keras.losses.BinaryCrossentropy()
learning_rate = 0.0001
input_shape = (224, 224, 3)


include_top = False
# metrics=["accuracy"] #, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
early_stopping_patience = 15
shuffle = True
class_mode = 'binary'  # 'categorical'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
trainable = False
training = False
dropout = 0.2



def save_history(name,history_df):

    history_excel_path = (
            save_path +
            name +
            '_history'
            '.xlsx'
    )

    history_df.to_excel(history_excel_path, index=False)
    print("Training history saved to:", history_excel_path)


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
#%%
vgg = tf.keras.applications.vgg16.VGG16(include_top=include_top,
                                                 weights="imagenet",
                                                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                                                # pooling="max",  # pooling mode for feature extraction
                                                 classes=2)

def make_vgg16_model(base_model: tf.keras.applications.vgg16.VGG16,) -> tf.keras.Model:

    # freeze base model
    base_model.trainable = False#trainable

    inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    x = base_model(inputs, training=training)
    x = tf.keras.layers.Flatten()(x)

    x = Dense(2048, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)

    output = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=METRICS,
    )

    return model






new_model = make_vgg16_model(base_model=vgg)

new_model.summary()
#%%
"""import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
tf.keras.utils.plot_model(new_model,
                          show_shapes=True,  # show_dtype=False,
                          show_layer_names=True,
                          layer_range=None,
                          show_layer_activations=True)"""
#%%
early_stopping = EarlyStopping(monitor="val_cohen_kappa",
                               verbose=1,
                               patience=early_stopping_patience,
                               mode="max",
                               baseline=0.0,
                               restore_best_weights=True)





for target_category in target_categories:
    train_file = "C:/Users/raouf/Desktop/pfe/datasets/" + target_category + "_df_train.csv"
    val_file = "C:/Users/raouf/Desktop/pfe/datasets/" + target_category + "_df_valid.csv"


    save_path = main_save_path + target_category + "/"
    model_name = architecture + "_" + target_category
    model_path = save_path + '_best_' + model_name + '.keras'
    model_checkpoint = ModelCheckpoint(filepath=model_path,
                                       monitor="val_cohen_kappa",
                                       save_best_only=True)
    # %%

    old_base_path = 'C:\\Users\\HP\\Documents\\ML\\encadrement Master\\MURA-V1.1\\'

    df_train = pd.read_csv(train_file)
    df_train['image_path'] = df_train['image_path'].str.replace(old_base_path, mura_location)

    df_valid = pd.read_csv(val_file)
    df_valid['image_path'] = df_valid['image_path'].str.replace(old_base_path, mura_location)

    df_test = load_data(test_file, target_category)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,
                                                              horizontal_flip=True,
                                                              fill_mode="constant",
                                                              cval=0.0,
                                                              preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
    train_generator = datagen.flow_from_dataframe(df_train,
                                                  directory=None,
                                                  x_col='image_path',
                                                  y_col='label',
                                                  target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                  batch_size=batch_size,
                                                  class_mode=class_mode,
                                                  )

    validation_generator = datagen.flow_from_dataframe(df_valid,
                                                       directory=None,
                                                       x_col='image_path',
                                                       y_col='label',
                                                       target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                       batch_size=batch_size,
                                                       class_mode=class_mode,
                                                       shuffle=False
                                                       )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
    test_generator = test_datagen.flow_from_dataframe(df_test,
                                                      directory=None,
                                                      x_col='image_path',
                                                      y_col='label',
                                                      target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                      batch_size=batch_size,
                                                      class_mode=class_mode,
                                                      shuffle=False
                                                      )
    # %%
    history = new_model.fit(train_generator,
                            # teps_per_epoch=len(df_train) // batch_size,
                            validation_data=validation_generator,
                            # validation_steps=len(df_valid) // batch_size,
                            epochs=num_epochs,
                            shuffle=shuffle,
                            callbacks=[early_stopping, model_checkpoint])  # , reduce_lr_callback])

    # %%
    new_model.save(save_path + model_name + '.keras')

    plot_metrics(history)

    new_model.evaluate(test_generator)

    y_pred_prob = new_model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_test = test_generator.classes
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    # %%

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create ConfusionMatrixDisplay object
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(4, 6))  # Adjust the figure size as needed
    cm_display.plot(cmap=plt.cm.Blues)  # You can choose any colormap you prefer
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.grid(False)  # Turn off grid
    plt.savefig(save_path + model_name + "_confusion_matrix.jpg")
    plt.show()

    # %%
    history_df = pd.DataFrame(history.history)
    save_history(model_name, history_df)

    new_model.save(save_path + model_name + "_pt_imagenet-{new_model.count_params()}.h5")
    vgg.save(save_path + model_name + ".h5")

    vgg.trainable = True

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
    # %%
    history = new_model.fit(train_generator,
                            # steps_per_epoch=len(df_train) // batch_size,
                            validation_data=validation_generator,
                            # validation_steps=len(df_valid) // batch_size,
                            epochs=fine_tuning_epochs,
                            shuffle=shuffle,
                            callbacks=[early_stopping, model_checkpoint, reduce_lr_on_plateau])

    # %%
    new_model.save(save_path + model_name + '.keras')
    plot_metrics(history)
    # %%

    new_model.evaluate(test_generator)

    # Make predictions using the test generator
    y_pred_prob = new_model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_test = test_generator.classes
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    # %%
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create ConfusionMatrixDisplay object
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(4, 6))  # Adjust the figure size as needed
    cm_display.plot(cmap=plt.cm.Blues)  # You can choose any colormap you prefer
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.grid(False)  # Turn off grid
    plt.savefig(save_path + model_name + "_confusion_matrix.jpg")
    plt.show()
    # %%
    history_df = pd.DataFrame(history.history)
    save_history(model_name, history_df)