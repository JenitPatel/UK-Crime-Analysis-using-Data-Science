import pandas as pd
import numpy as np

crime = pd.read_csv("D:/Data Mining Data/2014-2016.csv")
crime.columns = ['Crime ID', 'Month', 'Reported by', 'Falls within', 'Longitude', 'Latitude', 'Location', 'LSOA code', 'LSOA name', 'Crime type', 'Last outcome category', 'Context']
crime.pop('Crime ID')
print(crime.head())

grouped_by_lsoa = crime.groupby(["LSOA code"])
lsoa_total_crime = grouped_by_lsoa.size()
lsoa_total_crime.index.names = ['LSOA code']
lsoa_total_crime = lsoa_total_crime.reset_index()
lsoa_total_crime.columns = ['LSOA code', 'Total']
lsoa_total_crime = lsoa_total_crime.fillna(0)
lsoa_total_crime = pd.merge(lsoa_total_crime, grouped_by_lsoa["LSOA code", "Longitude", "Latitude"].first(), on="LSOA code")

lsoa_agg_data = crime.groupby(["LSOA code", "Crime type"]).size().unstack(level=-1)
lsoa_agg_data = lsoa_agg_data.fillna(0)
lsoa_agg_data.index.names = ['LSOA code']
lsoa_agg_data = lsoa_agg_data.reset_index()
lsoa_agg_data = pd.merge(lsoa_agg_data, lsoa_total_crime, on="LSOA code")
print(lsoa_agg_data)

import matplotlib.pyplot as plt
lsoa_agg_data.hist(bins=50, figsize=(20,15))
plt.show()

print(lsoa_agg_data["Bicycle theft"].max())

lsoa_agg_data.hist(bins=50, figsize=(20,15), range=(0, 1000))
plt.show()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(lsoa_agg_data, test_size=0.2, random_state=42)

print(len(train_set), "train +", len(test_set), "test")

crime = train_set.copy()

crime.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1, s=1)

crime.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1, s=5,
          c=np.log(crime["Total"]), cmap=plt.get_cmap("jet"), colorbar=True)

crime.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1, s=5,
          c=np.log(crime["Drugs"]), cmap=plt.get_cmap("jet"), colorbar=True)

crime.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1, s=5,
          c=np.log(crime["Vehicle crime"]), cmap=plt.get_cmap("jet"), colorbar=True)

crime.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1, s=5,
          c=np.log(crime["Violence and sexual offences"]), cmap=plt.get_cmap("jet"), colorbar=True)


corr_matrix = crime.corr()
corr_matrix["Total"].sort_values(ascending=False)

from pandas.tools.plotting import scatter_matrix

attributes = ["Anti-social behaviour", "Burglary", "Criminal damage and arson", "Other theft", "Shoplifting", "Total"]
scatter_matrix(crime[attributes], figsize=(12, 8))


