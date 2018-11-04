import matplotlib.pyplot as plot
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd
import numpy as np

df = pd.read_excel('titanic.xls')
print(df.head())
df.drop(['name','body'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def convert_to_numeric(df):
    cols = df.columns.values
    for col in cols:
        text_digit_dic = {}
        def get_int_of_text(text):
            return text_digit_dic[text]
        if df[col].dtype!=np.int64 and df[col].dtype!=np.float64:
            unique_contents = set(df[col].values.tolist())
            x = 0
            for unique in unique_contents:
                if unique not in text_digit_dic:
                    text_digit_dic[unique] = 0
                    x+=1
            df[col] = list(map(get_int_of_text, df[col]))
    return df;

df = convert_to_numeric(df)
print(df.head())
