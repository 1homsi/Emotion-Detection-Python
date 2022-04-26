import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import csv

df1 = pd.read_csv('./data/data1.csv')


plt.xlabel("Year")
plt.ylabel("Per Capita Income")
plt.scatter(df1.year, df1.pci)

reg1 = linear_model.LinearRegression()
reg1.fit(df1[['year']], df1.pci)

reg1.predict([[2021]])

plt.xlabel("Year")
plt.ylabel("Per Capita Income")
plt.scatter(df1.year, df1.pci)
plt.plot(df1.year, reg1.predict(df1[['year']]), color='red')