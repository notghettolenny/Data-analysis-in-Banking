import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
plt.rcParams["figure.figsize"] = (8, 6)

import warnings
warnings.filterwarnings('ignore')

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        pd.set_option("precision", 2)
pd.options.display.float_format = '{:.2f}'.format
#load dataset
train = pd.read_csv("/kaggle/input/banking-dataset-marketing-targets/train.csv", sep = ';')
test = pd.read_csv("/kaggle/input/banking-dataset-marketing-targets/test.csv", sep = ';')

train.head()
train.shape , test.shape
train.isna().sum() , test.isna().sum()
d = {"no": 0, "yes": 1}
train["y"] = train["y"].map(d)
train.columns
train["age"].hist()
sns.countplot(train["y"]);
sns.countplot(train["marital"])
train[["age", "marital"]].groupby("marital").mean().plot();
train[["age", "marital"]].groupby(
    "marital"
).mean().plot(kind="bar", rot=45);
sns.pairplot(train[["age", "duration", "campaign"]]);
sns.distplot(train.age);
sns.jointplot(x="age", y="duration", data=train, kind="scatter")
top_jobs = (train.job.value_counts().sort_values(ascending=False).head(5).index.values)
sns.boxplot(y="job", x="age", data=train[train.job.isin(top_jobs)], orient="h")
job_marital_y = (train.pivot_table(index="job", columns="marital", values="y", aggfunc=sum))
sns.heatmap(job_marital_y, annot=True, fmt="d", linewidths=0.5);age_df = (train.groupby("age")[["y"]].sum().join(train.groupby("age")[["y"]].count(), rsuffix='_count'))
age_df.columns = ["Attracted", "Total Number"]

trace0 = go.Scatter(x=age_df.index, y=age_df["Attracted"], name="Attracted")
trace1 = go.Scatter(x=age_df.index, y=age_df["Total Number"], name="Total Number")

data = [trace0, trace1]
layout = {"title": "Statistics by client age"}

fig = go.Figure(data=data, layout=layout)

iplot(fig, show_link=False)
month_index = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
month_df = (train.groupby("month")[["y"]].sum().join(train.groupby("month")[["y"]].count(), rsuffix='_count')).reindex(month_index)
month_df.columns = ["Attracted", "Total Number"]

trace0 = go.Bar(x=month_df.index, y=month_df["Attracted"], name="Attracted")
trace1 = go.Bar(x=month_df.index, y=month_df["Total Number"], name="Total Number")

data = [trace0, trace1]
layout = {"title": "Share of months"}

fig = go.Figure(data=data, layout=layout)

iplot(fig, show_link=False)
data = []

for status in train.marital.unique():
    data.append(go.Box(y=train[train.marital == status].age, name=status))
iplot(data, show_link=False)
