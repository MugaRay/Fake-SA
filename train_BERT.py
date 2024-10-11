import os

from models import create_Bert
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # supressing debugging info for tensorflow

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras import backend as K
import csv
import gc


def save_scores(name, scores):
    path = os.path.join(os.getcwd(),"BERT results", f"{name}_results.csv")
    with open(path, 'w', newline='') as file:
        field_names = ["Accuaracy", "F1", "Recall", "Precision", "AUC"]
        writer = csv.DictWriter(file, fieldnames=field_names)

        writer.writeheader()
        for i in scores:
            writer.writerow({"Accuaracy":i[1], "F1":i[4], "Recall":i[3], "Precision":i[2], "AUC": i[5]})


def getFilename(x):
    return os.path.splitext(os.path.basename(x))[0] 


def train_model(Train, X, y, out=None, mode=1):
    
    if out is None:
        out = f"{getFilename(Train)}"        
    
    # importing the dataset

    print(f"Training for: {out}")
    df = pd.read_csv(Train)

    
    X = df[X]
    y = df[y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if mode == 1:
        verbose = 1
        amount = 3
        epochs = 3
    else:
        verbose= 0
        amount = 5
        epochs = 10
    
    scores = []
    Best_model = None
    best_score = 0

    for _ in range(amount):
        model = create_Bert()

        model.fit(
            X_train,
            y_train,
            validation_split=0.4,
            epochs=epochs,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
            verbose=verbose,
            batch_size=16
        )
        
        score = model.evaluate(X_test, y_test)
        
        # Check for the highest score
        if score[1] > best_score:
            Best_model = model
            best_score = score[1]
        else:
            del model
            K.clear_session()  # Clear the Keras session
        
        scores.append(score)

    path = os.path.join(os.getcwd(),"BERT models", f"{out}.keras")   
    Best_model.save(path)
    save_scores(out, scores)
    gc.collect()  # Explicitly call garbage collection




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