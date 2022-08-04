import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd

#open excel/dataset sheet
#The name of my datasheet was final balanced dataset
df = pd.read_csv("FinalBalancedDataset.csv")
df.head(5)

#find amount of samples for 1 category
df_toxic = df[df['Toxicity'] == 1]

#find amount of samples for other category
df_nonetoxic = df[df['Toxicity'] == 0]

#downsample
df_nonetoxic_downsampled = df_nonetoxic.sample(df_toxic.shape[0])

#You may have to downsample further if it takes too long

#concatenate into one dataset
df_balanced = pd.concat([df_toxic, df_nonetoxic_downsampled])

#split the data into train and test of equal distribution of toxic and non-toxic tweets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_balanced['tweet'],df_balanced['Toxicity'], stratify=df_balanced['Toxicity'])

#download preprocessor and encoder
preprocess_url='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
bert_preprocess = hub.KerasLayer(preprocess_url)
bert_encoder = hub.KerasLayer(encoder_url)

#inputs sentences, outputs encoded text
def get_sentence_embedding(sentences):
    text_processed = bert_preprocess(sentences)
    return bert_encoder(text_processed)['pooled_output']

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
print(model.summary())

#Metrics
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
]

#Compile model
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics=METRICS)
#fit model
model.fit(X_train, y_train, epochs=10)

#evaluate model
model.evaluate(X_test, y_test)

#predict
y_predicted = model.predict(X_test)
y_predicted=y_predicted.flatten()


#round up or down
import numpy as np
y_predicted = np.where(y_predicted > 0.5, 1, 0)
print(y_predicted)

#confusion matrix (left top and right bottom are correct, else are incorrect)
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_predicted)
print(cm)

#classification report
print(classification_report(y_test, y_predicted))

#test tweets
tweets = [
    'I hate americans and I want them to die'
    'I just ate a donut!'
    'Im part of a cult the eats humans!'
    'you ever been to walmart past 9pm? its scary'
]

ans = model.predict(tweets)
print(ans)
