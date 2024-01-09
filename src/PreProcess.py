import pandas as pd
import numpy as np


df = pd.read_csv('LBW_Dataset.csv')
def data_cleaning(data):
  data['Weight'] = data.Weight.replace(np.nan,np.mean(data.Weight))
  data['Delivery phase']= data['Delivery phase'].interpolate('linear')
  data['HB'] = data['HB'].replace(np.nan,np.mean(data.HB))
  data['BP'] = data['BP'].replace(np.nan,np.mean(data.BP))
  data['Education'] = data['Education'].replace(np.nan,5)
  data['Residence']= data['Residence'].interpolate('linear')
  data['Age'] = data['Age'].replace(np.nan,round(np.mean(data.Age)))
  return data
data = data_cleaning(df)
data.to_csv('PreProcessed_LBW.csv',index=False)