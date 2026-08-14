import pandas as pd
import pingouin as pg

data = pd.read_csv("/home/simran/SLU/results/openended/interannotator.csv")

icc = pg.intraclass_corr(
    data=data, targets='sample', raters='rater', ratings='score'
).round(4)

print(icc)