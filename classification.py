import pandas as pd
from scipy.io import arff

# load the ARFF file and convert it to pandas dataframe 
data, meta = arff.loadarff('TimeBasedFeatures-Dataset-120s-AllinOne.arff')
df = pd.DataFrame(data)

# ARFF files often encode strings as bytes, this fixes that
df = df.apply(lambda col: col.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))

# take a look at the data 
print(df.shape)          # rows and columns
print(df.head())         # first 5 rows
print(df.columns.tolist())
print(df['class1'].value_counts())
print(df.isnull().sum())  # check for missing values
