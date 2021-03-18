import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from typing import List

class Bootstrap:
    values:np.array
    iterations:int
    sample_size:int

    def __init__(self, vals:np.array, iterations:int = 1000, percentage:float = 0.5):
        self.reset()
        self.values = vals
        self.iterations = iterations
        total_samples = len(vals)
        self.sample_size = int(total_samples*percentage)


    def reset(self):
        print("Bootstrap.reset()")
        self.values = np.array([])
        self.iterations = 1000
        self.sample_size = 0

    def run(self, num_iterations:int = -1, sample_size:int = -1):
        if num_iterations > 0:
            self.iterations = num_iterations
        if sample_size > 0:
            self.sample_size = sample_size

        stats = []
        for i in range(self.iterations):
            # prepare test/train sets
            train:np.array = resample(self.values, n_samples=self.sample_size)
            test = np.array([x for x in self.values if x.tolist() not in train.tolist()])
            # fit model
            model = DecisionTreeClassifier()
            model.fit(train[:,:-1], train[:,-1])
            # evaluate model
            predictions = model.predict(test[:,:-1])
            score = accuracy_score(test[:,-1], predictions)
            print(score)
            stats.append(score)

        # plot scores
        pyplot.hist(stats)
        pyplot.show()

        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))



def main():

    # get the data
    df = pd.read_csv('pima-indians-diabetes.data.csv', header=None)
    values = df.values
    print(values)

    b = Bootstrap(values)
    b.run()

if __name__ == "__main__":
    main()


