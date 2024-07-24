### Human Emotion Prediction with RoBERTa & XLNet: ###

This repository covers the code for performing text classification on the Human Emotion dataset 
using state-of-the-art Auto Encoder model RoBERTa & XLNet Auto Regressive model.
Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.
RoBERTa is a state-of-the-art transformer model used to perform complex NLP tasks like NLU, NLG, Sentiment Analysis,
Text Classification... and surpassed BERT in terms of performance metrics in these tasks.
XLNET is an Auto Regressive model based on the Transformer XL architecture which has found to exceed BERT performance
based on it's unique training technique.

Below are the steps to be followed:

1. Install the required packages stated in requirements.txt file 
   Packages can be installed on an Anaconda environment or on normal python interpreter.
   For Anaconda:
   conda create --name <youenvname>
   conda activate <yourenvname>
   pip install -r requirements.txt
   For Python Interpreter:
   pip install -r requirements.txt
   
2. The entire repository is modularised in to individual sections which performs specific task.
   First, go to src folder.
   Under src, there are 2 primary packages:
   a> ML_Pipeline:
   This contains individual modules with different function declarations to perform specific Machine Learning task.
   b> engine.py:
   This is the heart of the project, as all the function calls are done here.
   
3. Run/Debug the engine.py file and all the steps will be automatically taken care as per the logic.

4. All input datasets are stored in the input folder.

5. All predictions and models are stored in the output folder.