import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from collections import defaultdict
import re
import sys
import pdb 
import datetime

### Read the tweets data
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
data = pd.read_csv('./data/tweets.csv', sep="\t", parse_dates=['created_at'], date_parser=dateparse, dtype={'hashtags':str})
## print("Read the data. Its columns are:\n   " , re.sub("[ ]+",":", str(data.dtypes).replace("\n", ",\t") ) )


## Also create time-column at yy and yymm granularities 
##data['yy'] = data['created_at'].map(lambda x: x.year )
data['yymm'] = data['created_at'].map(lambda x: datetime.datetime(x.year,x.month,1,1,1)) #rounding it to the first day of the month
data['yy'] = data['created_at'].apply(lambda dt: datetime.datetime(dt.year, 1, 1, 1, 1, 1 ) )

#print(starting_year, data.yymm.head() , data.yy.head() )



## 1. Get tweet frequency spread over time 
fig, axes = plt.subplots(figsize=(15,6))
### yearly_counts = data['yy'].value_counts().reset_index().sort_values(by="index").set_index("index")
monthly_counts = data['yymm']
##monthly_counts.value_counts().reset_index().sort_values(by="index").set_index("index")
monthly_counts.value_counts().plot(ax=axes, kind='bar', sort_columns=False)
plt.title( "Tweet frequency over Time ", fontsize=14, y=1.0)
axes.set_ylabel("Number of Tweets ", fontsize=14)
axes.set_xlabel("Time (Each bar denotes number of tweets in a given month)", fontsize=14)
axes.set_xlabel("Date ", fontsize=14)
fig.tight_layout()
plt.savefig("data/tweet_frequency_monthly.jpg" )


## 2. Get tweet frequency spread over user-space 
fig, axes = plt.subplots(figsize=(15,6))
data['poster_name'].value_counts().plot(ax=axes, kind='bar')
#plt.setp(axes[0].get_xticklabels(), visible=False)
plt.title( "Tweet frequency over user-space ", fontsize=14, y=1.0)
axes.set_ylabel("Number of Tweets ", fontsize=14)
axes.set_xlabel("Time (Each bar denotes number of tweets made by a user)", fontsize=14)
axes.set_xlabel("User ", fontsize=14)
fig.tight_layout()
plt.savefig("data/tweet_frequency_user_wise.jpg" )



## 3. Get follower count for users
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
user_data = pd.read_csv('./data/tweet_users.csv', sep="\t")#, parse_dates=['created_at'], date_parser=dateparse, dtype={'hashtags':str})
#print("Read the data. Its columns are:\n   " , re.sub("[ ]+",":", str(user_data.dtypes).replace("\n", ",\t") ) )
fig, axes = plt.subplots(figsize=(15,5))
user_data.set_index('name')[['followers_count']].plot(ax=axes, kind='bar')
##plt.setp(axes[0].get_xticklabels(), visible=False)
plt.title( "Visualizing Tweet frequency: Year wise, and User-wise", fontsize=16)
axes.set_xlabel("Twitter user name", fontsize=16)
fig.tight_layout()
plt.savefig("data/followers.jpg")
