
import pandas as pd

read_file = pd.read_csv("data/probing_dataset.txt", sep=";")
print(read_file.dtypes)
print(read_file.info)
# read_file.to_csv("data/probing_dataset.csv", index=None)
