import pandas as pd
data = pd.read_csv('data2/Turbine_Data.csv')
df = pd.DataFrame(data)
df.dropna(inplace=True)
df.to_csv('./handle_Turbine_Data.csv', index=False)