import os 
from train_LSTM import train_model

dataset_dir = os.path.join(os.getcwd(), "Datasets")

datasets  = os.listdir(dataset_dir)
dataset_list = [os.path.join(os.getcwd(), "Datasets", i) for i in datasets]


def filter_csv_files(file_list):

  csv_files = []
  for file_path in file_list:
    if file_path.endswith(".csv"):
      csv_files.append(file_path)
  return csv_files


dataset_list = filter_csv_files(dataset_list)


def train(x, mode=1):
  train_model(x, "data", "label", mode=mode)



for i in dataset_list:
    train(i, mode=2)