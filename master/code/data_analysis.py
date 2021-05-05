from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import json
length = 0
i = 0
data = defaultdict(int)
file = "data/uci_dataset_with_tags.txt"
with open(file, "r") as f:
    if file.endswith(".jl"):
        for line in f:
            game = json.loads(line)
            data[str(len(game["uci"].split()[2:]))] += 1
    else:  # txt
        for line in f:
            data[str(len(line.split()[2:]))] += 1

    num_games = sum(data.values())
    df = pd.DataFrame([(int(k), v) for k, v in data.items()], columns=["Zugnummer", "Anzahl"])
    df = df.sort_values(by='Zugnummer')
    print(num_games)
    print(df)
    plot = df.plot(x='Zugnummer', y='Anzahl')
    fig = plot.get_figure()
    fig.savefig("GPT_training_dataset.png")
    plt.show()
