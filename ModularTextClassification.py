import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd

def prepare_datasets():
    #open excel/dataset sheet
    df = pd.read_csv("FinalBalancedDataset.csv")

    #find amount of samples for 1 category
    df_toxic = df[df['Toxicity'] == 1]

    #find amount of samples for other category
    df_nonetoxic = df[df['Toxicity'] == 0]

    #downsample
    df_nonetoxic_downsampled = df_nonetoxic.sample(df_toxic.shape[0])

    #further downsampling
    df_nt_ds2 = df_nonetoxic_downsampled.sample(200)
    df_t_ds = df_toxic.sample(200)

    #concatenate into one dataset
    df_balanced = pd.concat([df_t_ds, df_nt_ds2])

    #split the data into train and test of equal distribution of toxic and non-toxic tweets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_balanced['tweet'],df_balanced['Toxicity'], stratify=df_balanced['Toxicity'])
    return X_train, X_test, y_train, y_test

#inputs sentences, outputs encoded text
def get_sentence_embedding(sentences):
    text_processed = bert_preprocess(sentences)
    return bert_encoder(text_processed)['pooled_output']

def build_model():

    #download preprocessor and encoder
    preprocess_url='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    encoder_url='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
    bert_preprocess = hub.KerasLayer(preprocess_url)
    bert_encoder = hub.KerasLayer(encoder_url)

    #functional tf model
    #Bert layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    #Neural network layers
    l = tf.keras.layers.Dropout(0.1, name='dropout')(outputs['pooled_output'])
    l = tf.keras.layers.Dense(1, activation='sigmoid', name = 'output')(l)

    #construct final model
    model = tf.keras.Model(inputs=[text_input], outputs=[l])
    return model

def predict(model, X_test):
    #predict
    y_predicted = model.predict(X_test)
    y_predicted=y_predicted.flatten()


    #round up or down
    import numpy as np
    y_predicted = np.where(y_predicted > 0.5, 1, 0)
    print(y_predicted)

if __name__ == "__main__":
    #create train and test sets
    X_train, X_test, y_train, y_test = prepare_datasets()

    #Build the network
    model = build_model()

    #Metrics for compilation
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]

    #Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer = optimizer,
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])
    
    #train the model
    model.fit(X_train, y_train, batch_size=5, epochs=10)

    #evaluate model on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    model.save("C:\\Users\\maxwh\\Desktop\\BERT\\Model")
    #to run "tensorflow.kera.models.load_model(path_of_saved_model)"
    #use "model.predict()" passing the data to predict
