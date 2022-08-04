# Toxic Tweets
Toxic statements

1) Preparing your datasets
If you want to make your own datasheet, ensure it is in the same format as this one. When you get to the modulartextclassification.py
file, change the words in lines 12 and 15 that say toxicity to whatever it is you are measuring. Ensure it the excel file is saved in the same 
folder as the rest of these files.

2) Training your model 
The modular text classification file trains and saves the Model inside a sub folder.
Variables you may want to alter include the name of the datasheet on line 9 and where the model is saved in line 97.
Once you have run this code all the way through once, you will not need it again, you can access your model from other codes.

3) Testing your model
The test file is a simple use case that gets the model and uses the predict function to input a statement and return the toxicity prediction
