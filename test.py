import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow.keras
text = input('Input tweet: ')
#input tweets here
text_test = [
    text
]
# load model
z = tensorflow.keras.models.load_model("C:\\Users\\maxwh\\Desktop\\BERT\\Model")
#output prediction
pred = z.predict(text_test)
#print prediction
print("Predicted Toxicity level: {}%".format(pred))

