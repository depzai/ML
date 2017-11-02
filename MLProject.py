import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import sklearn
import pandas as pd
import os
import tarfile

df = pd.read_csv("housing.csv")
print(df.head())