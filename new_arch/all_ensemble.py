from my_imports2 import *
from my_functions2 import *


mura_location = "C:/Users/raouf/Desktop/pfe/"
data_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/'
train_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/train'
test_path = 'C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid'





test_file = "C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid_image_paths.csv"

target_categories = ["XR_FINGER","XR_SHOULDER","XR_WRIST"]

#target_category = "XR_SHOULDER"



architecture = 'ensemble_all'

main_save_path = "C:/Users/raouf/Desktop/pfe/history/new_arch/"+architecture+"/"





batch_size = 16
num_epochs = 100

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

def save_history(name,history_df):

    history_excel_path = (
            save_path +
            name +
            '_history' 
            '.xlsx'
    )

    history_df.to_excel(history_excel_path, index=False)
    print("Training history saved to:", history_excel_path)



for target_category in target_categories:
    densenet_path = "C:/Users/raouf/Desktop/pfe/history/final/densenet/" + target_category + "/best/"
    vgg_path = "C:/Users/raouf/Desktop/pfe/history/final/vgg/" + target_category + "/best/"
    resnet_path = "C:/Users/raouf/Desktop/pfe/history/final/resnet/" + target_category + "/best/"

    files = glob.glob(os.path.join(densenet_path, "*"))
    densenet_path = files[0]
    densenet_model = load_model(densenet_path, custom_objects={'F1Score': F1Score})
    densenet_model.trainable = False
    densenet_base = densenet_model.get_layer(name="densenet201")

    files = glob.glob(os.path.join(vgg_path, "*"))
    vgg_path = files[0]
    vgg_model = load_model(vgg_path, custom_objects={'F1Score': F1Score})
    vgg_model.trainable = False
    vgg_base = vgg_model.get_layer(name="vgg16")

    files = glob.glob(os.path.join(resnet_path, "*"))
    resnet_path = files[0]
    resnet_model = load_model(resnet_path, custom_objects={"F1Score": F1Score})
    resnet_model.trainable = False
    resnet_base = resnet_model.get_layer(name="resnet50")





    def build_ensemble_model():
        inputs = tf.keras.Input(shape=input_shape)

        densenet_preprocess_input = preprocess_input_densenet(inputs)
        densenet_base_output = densenet_base(densenet_preprocess_input)

        vgg_preprocess_input = preprocess_input_vgg(inputs)
        vgg_base_output = vgg_base(vgg_preprocess_input)
        vgg_base_output = tf.keras.layers.Flatten()(vgg_base_output)

        resnet_preprocess_input = preprocess_input_resnet(inputs)
        resnet_base_output = resnet_base(resnet_preprocess_input)

        concat_output = tf.keras.layers.Concatenate(axis=-1)([resnet_base_output, vgg_base_output,densenet_base_output])

        x = Dense(2048, activation="relu")(concat_output)

        x = Dense(512, activation="relu")(x)

        x = Dense(64, activation="relu")(x)

        x = Dropout(0.2)(x)

        prediction = Dense(1, activation="sigmoid")(x)

        model = Model(inputs, prediction)
        model._name = "Ensemble_model"

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=METRICS)
        return model


    new_model = build_ensemble_model()

    new_model.summary()

    early_stopping = EarlyStopping(monitor="val_cohen_kappa",
                                   verbose=1,
                                   patience=early_stopping_patience,
                                   mode="max",
                                   baseline=0.0,
                                   restore_best_weights=True)

    save_path = main_save_path + target_category + "/"
    model_name = architecture + "_" + target_category
    model_path = save_path + '_best_' + model_name + '.keras'
    model_checkpoint = ModelCheckpoint(filepath=model_path,
                                       monitor="val_cohen_kappa",
                                       save_best_only=True)

    train_file = "C:/Users/raouf/Desktop/pfe/datasets/" + target_category + "_df_train.csv"
    val_file = "C:/Users/raouf/Desktop/pfe/datasets/" + target_category + "_df_valid.csv"

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
                                                       class_mode=class_mode,
                                                       shuffle=False
                                                       )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_generator = test_datagen.flow_from_dataframe(df_test,
                                                      directory=None,
                                                      x_col='image_path',
                                                      y_col='label',
                                                      target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                      batch_size=batch_size,
                                                      class_mode=class_mode,
                                                      shuffle=False
                                                      )

    history = new_model.fit(train_generator,
                            # teps_per_epoch=len(df_train) // batch_size,
                            validation_data=validation_generator,
                            # validation_steps=len(df_valid) // batch_size,
                            epochs=num_epochs,
                            shuffle=shuffle,
                            callbacks=[early_stopping, model_checkpoint])  # , reduce_lr_callback])

    new_model.save(save_path + model_name + '.keras')

    plot_metrics(history)

    new_model.evaluate(test_generator)

    y_pred_prob = new_model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_test = test_generator.classes
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

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

    history_df = pd.DataFrame(history.history)
    save_history(model_name, history_df)

    new_model.save(save_path + model_name + "_pt_imagenet-{new_model.count_params()}.h5")
    new_model.save(save_path + model_name + ".h5")

    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
