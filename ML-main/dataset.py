import pandas as pd 
import random
rows=20
cols=5
matrix=[[random.randint(0,100) for _ in range(cols)] for _ in range(rows)] 
df=pd.DataFrame(matrix)
#print(df)
filename='dataset.csv'
df.to_csv(filename,index=False)
print(filename)