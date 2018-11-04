from __future__ import division
import matplotlib.pyplot as plot
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd
import numpy as np

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
def convert_to_numeric(df):
    cols = df.columns.values
    for col in cols:
        text_digit_dic = {}
        def get_int_of_text(text):
            return text_digit_dic[text]
        if df[col].dtype!=np.int64 and df[col].dtype!=np.float64:
            col_values = df[col].values.tolist()
            unique_contents = set(col_values)
            x = 0
            for unique in unique_contents:
                if unique not in text_digit_dic:
                    text_digit_dic[unique] = x
                    x+=1
            df[col] = list(map(get_int_of_text, df[col]))
    return df;
df = convert_to_numeric(df)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = KMeans(n_clusters=2)
clf.fit(X)
correct = 0

for i in range(len(X)):
    prediction_set = np.array(X[i].astype(float))
    prediction_set = prediction_set.reshape(-1, len(prediction_set))
    prediction_result = clf.predict(prediction_set)
    if prediction_result[0] == y[i]:
        correct += 1
if (correct/len(X))<0.5:
    print('accuracy', 1-(correct/len(X)))
else:
    print(correct/len(X))