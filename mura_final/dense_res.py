
from my_imports import *
from my_functions import *

def save_history(name,history_df):
    history_excel_path = (
            save_path +
            name +
            '_history' 
            '.xlsx'
    )
    history_df.to_excel(history_excel_path, index=False)
    print("Training history saved to:", history_excel_path)

main_save_path = "C:/Users/raouf/Desktop/pfe/history/all_prof_mura/dense_res/"
mura_location = "C:/Users/raouf/Desktop/pfe/"
data_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/'
train_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/train'
test_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid'
train_file = "C:/Users/raouf/Desktop/pfe/MURA-v1.1/train_image_paths.csv"
val_file = "C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid_image_paths.csv"


"""     , 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER',  'XR_WRIST'  """
target_categories = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER',  'XR_WRIST']
target_category = 'XR_HUMERUS'
batch_size = 8
num_epochs = 150
fine_tuning_epochs =1
validation_split = 0.2
loss_function = tf.keras.losses.BinaryCrossentropy()
learning_rate = 0.0001
input_shape = (224, 224, 3)
architecture = 'DenseRes'

include_top = False
# metrics=["accuracy"] #, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
early_stopping_patience = 25
shuffle = True
class_mode = 'binary'  # 'categorical'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
trainable = False
training = False
dropout = 0.2






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




"""tf.keras.utils.plot_model(new_model,
                          show_shapes=True,  # show_dtype=False,
                          show_layer_names=True,
                          layer_range=None,
                          show_layer_activations=True)"""

# Set training callbacks
early_stopping = EarlyStopping(monitor="val_cohen_kappa",
                               verbose=1,
                               patience=early_stopping_patience,
                               mode="max",
                               baseline=0.0,
                               restore_best_weights=True)




reduce_lr_callback = ReduceLROnPlateau(monitor='val_cohen_kappa',
                                       factor=0.9,
                                       patience=15,
                                       min_lr=0.0001)

for target_category in target_categories:

    densenet_path = "C:/Users/raouf/Desktop/pfe/history/all_prof_mura/densenet/" + target_category + "/best/"
    files = glob.glob(os.path.join(densenet_path, "*"))
    densenet_path = files[0]
    densenet_model = load_model(densenet_path, custom_objects={'F1Score': F1Score})
    densenet_model.trainable = False
    densenet_base = densenet_model.get_layer(name="densenet201")

    resnet_path = "C:/Users/raouf/Desktop/pfe/history/all_prof_mura/resnet/" + target_category + "/best/"
    files = glob.glob(os.path.join(resnet_path, "*"))
    resnet_path = files[0]
    resnet_model = load_model(resnet_path, custom_objects={"F1Score": F1Score})
    resnet_model.trainable = False
    resnet_base = resnet_model.get_layer(name="resnet50")

    ensemble_model = build_ensemble_model()

    ensemble_model.summary()

    save_path = main_save_path + target_category + "/"
    model_name = architecture + "_" + target_category


    model_path = save_path + '_best_' + model_name + '.keras'
    model_checkpoint = ModelCheckpoint(filepath=model_path,
                                       monitor="val_cohen_kappa",
                                       save_best_only=True)

    df_train = load_data(train_file, target_category)
    df_test = load_data(val_file, target_category)

    df_train, df_valid = train_test_split(df_train,
                                          test_size=validation_split,
                                          random_state=42,
                                          shuffle=True)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,
                                                              horizontal_flip=True,
                                                              fill_mode="constant",
                                                              cval=0.0,
                                                              )
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
                                                       class_mode=class_mode
                                                       )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_generator = test_datagen.flow_from_dataframe(df_test,
                                                      directory=None,
                                                      x_col='image_path',
                                                      y_col='label',
                                                      target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                      batch_size=batch_size,
                                                      class_mode=class_mode
                                                      )




    history = ensemble_model.fit(train_generator,
                                 # teps_per_epoch=len(df_train) // batch_size,
                                 validation_data=validation_generator,
                                 # validation_steps=len(df_valid) // batch_size,
                                 epochs=num_epochs,
                                 shuffle=shuffle,
                                 callbacks=[early_stopping, model_checkpoint])  # , reduce_lr_callback])

    ensemble_model.save(save_path + model_name + '.keras')

    plot_metrics(history)

    ensemble_model.evaluate(validation_generator)
    ensemble_model.evaluate(test_generator)

    # Make predictions using the validation generator
    y_valid_pred_prob = ensemble_model.predict(validation_generator)
    y_valid_pred = (y_valid_pred_prob > 0.5).astype(int)
    y_valid = validation_generator.classes
    print("Classification Report:")
    print(classification_report(y_valid, y_valid_pred))

    y_pred_prob = ensemble_model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_test = test_generator.classes
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cohen_kappa_score(y_test, y_pred)

    plt.rcParams.update({'font.size': 18})
    conf_matrix = confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                                display_labels=test_generator.class_indices.keys())
    cm_display.plot()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path + model_name + "_confusion_matrix.jpg")
    plt.show()


    # history_file = save_path + model_name + '_history'
    history_df = pd.DataFrame(history.history)
    save_history(model_name, history_df)

    ensemble_model.save(save_path + model_name + f"_pt_imagenet-{ensemble_model.count_params()}.h5")
    ensemble_model.save(save_path + model_name + ".h5")

"""
    ensemble_model.trainable = True

    ensemble_model.compile(
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
                                       save_best_only=True,
                                       overwrite=True)

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_cohen_kappa",
        verbose=1,
        patience=4,
        mode="max",
        factor=0.2,
        min_lr=1e-10,
    )

    history = ensemble_model.fit(train_generator,
                                 # steps_per_epoch=len(df_train) // batch_size,
                                 validation_data=validation_generator,
                                 # validation_steps=len(df_valid) // batch_size,
                                 epochs=fine_tuning_epochs,
                                 shuffle=shuffle,
                                 callbacks=[early_stopping, #model_checkpoint, reduce_lr_on_plateau
                                            ])

    ensemble_model.save(save_path + model_name + '.keras',overwrite=True)
    plot_metrics(history)

    ensemble_model.evaluate(validation_generator)
    ensemble_model.evaluate(test_generator)

    # Make predictions using the validation generator
    y_valid_pred_prob = ensemble_model.predict(validation_generator)
    y_valid_pred = (y_valid_pred_prob > 0.5).astype(int)
    y_valid = validation_generator.classes
    print("Classification Report:")
    print(classification_report(y_valid, y_valid_pred))

    # Make predictions using the test generator
    y_pred_prob = ensemble_model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_test = test_generator.classes
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cohen_kappa_score(y_test, y_pred)

    plt.rcParams.update({'font.size': 18})
    conf_matrix = confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                                display_labels=test_generator.class_indices.keys())
    cm_display.plot()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path + model_name + "_confusion_matrix.jpg")
    plt.show()
    # history_file = save_path + model_name + '_history'
    history_df = pd.DataFrame(history.history)
    save_history(model_name, history_df)"""
