import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # supressing debugging info for tensorflow

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from  models import Marivate_model
import csv
from pipeline import clean_text
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from keras import backend as K


def save_model(name, model):
    path = os.path.join(os.getcwd(),"LSTM models", f"{name}.weights.h5")
    model.save_weights(path, overwrite=True)
    


def add_to_CSV(name, scores):
    path = os.path.join(os.getcwd(),"LSTM results", f"{name}_results.csv")
    with open(path, 'w', newline='') as file:
        field_names = ["Accuaracy", "F1", "Recall", "Precision", "AUC"]
        writer = csv.DictWriter(file, fieldnames=field_names)

        writer.writeheader()
        for i in scores:
            writer.writerow({"Accuaracy":i[1], "F1":i[4], "Recall":i[3], "Precision":i[2], "AUC": i[5]})



def getFilename(x):
    return os.path.splitext(os.path.basename(x))[0] 

def train_model(Train, X, y, out=None, mode=1):

    """
    This function does the general training
    Everything can be broken up into smaller functions 
    mode : 1 testing, 2: evalyating models
    """

    if out is None:
        out = f"{getFilename(Train)}"        
    
    # importing the dataset

    print(f"Training for: {out}")
    df = pd.read_csv(Train)

    # cleaning dataset
    df.dropna(inplace=True)
    #print(df.shape)
    df["Text"] = df.apply(lambda row: clean_text(row[X], tweet=True) if row["type"] == "tweet" else clean_text(row[X]), axis=1)


    # getting the max length of a sentence in the dataset 
    full_text_length = df["Text"].apply(lambda x: len(x.split()))     # split it into indivdual words, then check

    max_length = full_text_length.max()

    X = df["Text"]
    y = df[y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    # TOKENIZOR 
    token = Tokenizer(num_words=max_length, oov_token="<OOV>")
    token.fit_on_texts(X_train)

    # text processing 
    training_seq = token.texts_to_sequences(X_train)
    testing_seq =  token.texts_to_sequences(X_test)

    #paqdding 
    train_padded = pad_sequences(training_seq,  maxlen=max_length,  padding='post')
    test_padded = pad_sequences(testing_seq, maxlen=max_length, padding='post')

   
    if mode == 1:
        verbose = 1
        amount = 3
        epochs = 3
    else:
        verbose= 0
        amount = 5
        epochs = 10



    scores_list = []
    Model = None
    batch_size = 16
    valid_split = 0.4
    Best_model = None
    best_score = 0

    callbacks = [
            EarlyStopping(monitor='val_loss',  patience=3, verbose=verbose)
        ]

    for _ in range(amount):
        Model = Marivate_model()

        Model.fit(train_padded, y_train,  batch_size=batch_size,  epochs=epochs,  validation_split=valid_split, callbacks=callbacks, verbose=verbose)
            

        score = Model.evaluate(test_padded, y_test, verbose=1)

        if score[1] > best_score:
            Best_model = Model
            best_score = score[1]
        else:
            del Model
            K.clear_session()  # Clear the Keras session

        scores_list.append(score)
        if mode ==1: break # ending for testing
        
        add_to_CSV(out, scores_list)
        save_model(out, Best_model)
        Model = None
        gc.collect()







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='Model Training',
                        description='Insert a dataset so the model can train it',
                        epilog='Text at the bottom of help')




    parser.add_argument('--train', required=True)
    parser.add_argument('--X', required=True)
    parser.add_argument('--Y', required=True)
    parser.add_argument('--out')
    parser.add_argument('--test')   

    args = parser.parse_args()
    train_model(args.train, args.X, args.Y, args.out)