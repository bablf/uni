
import pandas as pd
length = 0
i = 0
with open("data/uci_dataset_with_tags.txt", "r") as f:
    for line in f:
         length +=len(line.split()[1:])
         i+=1

    print(length)
    print(i)
    print(length/i)