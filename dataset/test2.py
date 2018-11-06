import pandas as pd
import numpy as np

df = pd.DataFrame([[4,2,3],[4,10,6],[7,8,5]],columns=['col1','col2','col3'])
print(df)
new=df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print (new)