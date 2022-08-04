# Toxic Tweets

1) Preparing your datasets -
If you want to make your own datasheet, ensure it is in the same format as this one. When you get to the 'modulartextclassification.py'
file, change the words in lines 12 and 15 that say toxicity to whatever it is you are measuring. If you have changed the name of the excel dataset from 'FinalBalancedDataset.csv', ensure you change that in the code on line 9 aswell. The excel file must be saved in the same folder as the rest of these files.

2) Training your model -
The 'modulartextclassification.py' file should be ran first, this trains and saves the Model inside a sub folder.
The model is saved in the directory stated in line 97, mine is for my personal file system so you will need to change that to fit you
Once you have run this code all the way through once, you will not need it again, you can access your model from other codes.

3) Testing your model -
The 'test.py' file can then be ran second, it is a simple use case that gets the model and uses the predict function to input a statement and return the toxicity prediction. You may need to change the directory if your model to load in line 7 to whereever you set it to save in line 97 of the previous file.
