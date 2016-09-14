import numpy as np
import pandas as pd

data_src = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'

df = pd.read_csv(data_src, header=None)

df.tail()

print(df)