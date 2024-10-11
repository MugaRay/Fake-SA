"""
Outputs the average of the models results in the results folders
"""
import os
import csv
from functools import reduce

LSTM_Folder =  os.path.join(os.getcwd(), "LSTM results")

BERT_Folder = os.path.join(os.getcwd(), "BERT results")

def fold(fn, state, lst):
    result = None
    for index, value in enumerate(lst):
        if index == 0:
            result = fn(state, value)
        else:
            result = fn(result, value)
    return result


def get_data(dataset):
    with open(dataset, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)  # Skip the header row

        avgFn = lambda state, v: [state[i] + float(v[i]) for i in range(len(state))]
        avg = fold(avgFn, [0,0, 0, 0], csvreader)
        results = [round(i /5, 4) *100 for i in avg ]
        return results


def generate_Latex_table(model_name, dataset, data):
    pass

def generate_table(model_name, dataset, data, cols=None):
    print(f"{model_name} trained on {dataset}")
    values = [f"{i}%" for i in data]
    if cols:
        value_str = map(lambda x: f"{x[0]}: {x[1]} ", zip(cols, values))
        combined = reduce(lambda x, y: x+y, value_str)
        print(combined)
    else:
        print(values)



if __name__ == "__main__":

    cols = ["Accuracy", "F1", "precision", "Recall"]
    LSTM_datasets = os.listdir(LSTM_Folder)
    Bert_datasets = os.listdir(BERT_Folder)

    for i in LSTM_datasets:
        path = os.path.join(LSTM_Folder, i)
        results = get_data(path)
        generate_table("LSTM", i, results, cols)
    
    for i in Bert_datasets:
        path = os.path.join(BERT_Folder, i)
        results = get_data(path)
        generate_table("BERT", i, results, cols)