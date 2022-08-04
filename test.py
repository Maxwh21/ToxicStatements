import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow.keras
#input tweets here
text = ['I hate people from Papua new guinea']
# load model
z = tensorflow.keras.models.load_model("C:\\Users\\maxwh\\Desktop\\BERT\\Model")
#output prediction
pred = z.predict(text)
#print prediction
print("Predicted Toxicity level: {}%".format(pred))
#the closer to 1 the output the more toxic
#the closer to 0 the less toxic
#due to the function used, if its above 0.5 its classed as toxic
#and below its classed as none toxic

