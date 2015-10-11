# AnalysisProgram.py,  Eric Titus,  October 1, 2015
# Read in the clustering analysis data, and process

# Import all relevant packages
from __future__ import print_function


import logging
import sys
from time import time

import numpy as np
import pandas as pd #pandas is for dataframes
import nltk  #natural language toolkit for stopwords and tokenizers
import re
import os as os
import codecs 	#deals with unicode files
import matplotlib.pyplot as plt  #plotting in matplotlib
import pickle  #use to save variables
# %matplotlib inline 

import seaborn as sns  #this makes the plots look good. 
sns.set_context("notebook", font_scale=1.5, 
	rc={"lines.linewidth": 2.5})

# import the dataframe
f=open('twtWlabledRTs.pckl','rb')
twtWlabledRTs=pickle.load(f)
f.close()

filename='TweetContent.txt'
baseFilename=re.split('[.]',filename)[0]

from gensim import models, corpora
# import the 3 structures from the gensim LDA analysis
# the lda model, the tfidf corpus, and the dictionary
lda=models.ldamodel.LdaModel.load(baseFilename + '.LDA')
mmTFIDF = corpora.MmCorpus(baseFilename+'TFIDF.mm')
dictionary=corpora.Dictionary.load(baseFilename + '.dict')

# this gives values showing how each post relates to each category
# For each post, there are n-topic values. The higher the value,
# the more the post matches that topic.

topicProbs=lda.inference(mmTFIDF)

topicArray=topicProbs[0]
topicArrayNorm=[]
for top in topicArray:
    topicArrayNorm.append(top/top.sum())

# pick out the two most likely topics for each post
labels1=[val.argsort()[-1] for val in topicArrayNorm]
labels2=[val.argsort()[-2] for val in topicArrayNorm]

# extract the likelihood of the post belonging to the 2 top topics.
labelRank1=[]
labelRank2=[]
for idx in np.arange(len(labels1)):
    labelRank1.append(topicArrayNorm[idx][labels1[idx]])
    labelRank2.append(topicArrayNorm[idx][labels2[idx]])

# stick these labels onto the dataframe
def AppendLabels(df,var,name):
	df.loc[:,name]=pd.Series(data=var,index=df.index)
	return(df)

twtWlabledRTs=AppendLabels(twtWlabledRTs,labels1,'Labels1')
twtWlabledRTs=AppendLabels(twtWlabledRTs,labels2,'Labels2')
twtWlabledRTs=AppendLabels(twtWlabledRTs,labelRank1,'LabelRank1')
twtWlabledRTs=AppendLabels(twtWlabledRTs,labelRank2,'LabelRank2')


numTopics=len(topicProbs[0][0])

#output the tweets that match each topic the best to text files

for idx in np.arange(numTopics):
	temp=(twtWlabledRTs.Content.ix
		[twtWlabledRTs.LabelRank1
		[twtWlabledRTs.Labels1==idx].nlargest(10).index].values)
	f=open('Topic' + str(idx) + 'top tweets.txt','w')
	for item in temp:
		f.write(item.encode('utf8')+'\n')
	f.close()


# plot data showing the number of UNIQUE Posts
# per category. RTs are not included
groupByLabels=twtWlabledRTs.Content.groupby(
	twtWlabledRTs.Labels1).count().plot(kind='bar')
plt.xlabel('Topic #')
plt.ylabel('Number of Posts')
plt.title('Unique Posts in each category')
plt.show()

# plot data showing the number of Posts
# per category. RTs ARE included
groupByLabels=twtWlabledRTs.RT_count.groupby(
	twtWlabledRTs.Labels1).sum().plot(kind='bar')
plt.xlabel('Topic #')
plt.ylabel('Number of Posts')
plt.title('Posts in each category, incl. RTs')
plt.show()

# Plot number of unique users generating unique content in each category
groupByLabelUser=twtWlabledRTs.Handle2.drop_duplicates().groupby(
	twtWlabledRTs.Labels1).count().plot(kind='bar')
plt.xlabel('Topic #')
plt.ylabel('user count')
plt.title('Users generating original content, by label')
plt.show()

# Plot a breakdown of when users are posting to twitter.
# unique content only (No RTs)
groupByHour=twtWlabledRTs.Handle2.groupby(
	twtWlabledRTs.Date.dt.hour).count().plot(kind='bar')
plt.xlabel('Hour of the day?')
plt.ylabel('unique post count')
plt.title('When are users posting unique content?')
plt.show()

# Plot a breakdown of when users are posting to twitter.
# Include RTs
groupByHour=twtWlabledRTs.RT_count.groupby(
	twtWlabledRTs.Date.dt.hour).sum().plot(kind='bar')
plt.xlabel('Hour of the day')
plt.ylabel('unique post count')
plt.title('When are users posting, including RTs?')
plt.show()

# I noticed a surge in RTs at 20:00 and 21:00. Is this topic specific?
groupByHour=twtWlabledRTs.loc[twtWlabledRTs.Date.dt.hour.isin(
	[20,21]),'RT_count'].groupby(
    twtWlabledRTs.Labels1).sum().plot(kind='bar')
plt.xlabel('Topic #')
plt.ylabel('post count')
plt.title('Do any topic surge with the increased RTs at 20:00 and 21:00?')
plt.show()

# how many unique posts are there each week day?
groupByHour=twtWlabledRTs.Handle2.groupby(
	twtWlabledRTs.Date.dt.dayofweek).count().plot(kind='bar')
plt.xlabel('Day of the Week')
plt.ylabel('unique post count')
plt.title('When are users posting unique content? \n Monday=0, Sunday=6')
plt.show()


# how many total posts (including RTs) are there each day?
groupByHour=twtWlabledRTs.RT_count.groupby(
	twtWlabledRTs.Date.dt.dayofweek).sum().plot(kind='bar')
plt.xlabel('Day of the Week')
plt.ylabel('post count')
plt.title('When are users posting any content? \n Monday=0, Sunday=6')
plt.show()

# Saturday showed a spike. When are users posting on Saturdays?
groupByHour=twtWlabledRTs.loc[twtWlabledRTs.Date.dt.dayofweek==5,
'RT_count'].groupby(twtWlabledRTs.Date.dt.hour).sum().plot(kind='bar')
plt.xlabel('Hour of Day')
plt.ylabel('unique post count')
plt.title('When are users posting any content? \n on Saturday?')
plt.show()

# What topics are posted about on Saturdays?
groupByHour=twtWlabledRTs.loc[twtWlabledRTs.Date.dt.dayofweek==5,'RT_count'].groupby(
	twtWlabledRTs.Labels1).sum().plot(kind='bar')
plt.xlabel('Topic number')
plt.ylabel('total post count')
plt.title('What topics are users posting about? \n on Saturday?')
plt.show()

# What topics are posted about on Saturday night?
groupByHour=twtWlabledRTs.loc[twtWlabledRTs.Date.dt.dayofweek==5 & twtWlabledRTs.Date.dt.hour.isin([18,19,20,21]) ,'RT_count'].groupby(twtWlabledRTs.Labels1).sum().plot(kind='bar')
plt.xlabel('Topic number')
plt.ylabel('total post count')
plt.title('What topics are users posting about? \n on Saturday night (18:00-22:00?')
plt.show()

# What topics are male users posting in?
groupByHour=twtWlabledRTs.loc[twtWlabledRTs.Gender=='male','RT_count'].groupby(twtWlabledRTs.Labels1).sum().plot(kind='bar')
plt.xlabel('Topic number')
plt.ylabel('total post count')
plt.title('number of all tweets by males per category')
plt.show()

# What topics are female users posting in?
groupByHour=twtWlabledRTs.loc[twtWlabledRTs.Gender=='female','RT_count'].groupby(twtWlabledRTs.Labels1).sum().plot(kind='bar')
plt.xlabel('Topic number')
plt.ylabel('total post count')
plt.title('number of all tweets by females per category')
plt.show()

# output the top unique users posting in each label category.
gbLabelsAndUsers=twtWlabledRTs.groupby(
	['Labels1','Handle2'])['Content'].count()

for idx in np.arange(numTopics):
	temp=gbLabelsAndUsers[idx].nlargest(10)
	f=open('Topic' + str(idx) + 'top unique posting users.txt','w')
	for entry in temp.index:
		string=entry + ' ' + str(temp[entry])
		f.write(string.encode('utf8')+'\n')
	f.close()

f=open('TopicKeywords.txt','w')
f.write('Keywords that define each topic \n')
for idx in np.arange(numTopics):
    prntStr='Topic ' + str(idx) + ': '
    for top in lda.show_topic(idx,topn=10):
        prntStr+=top[1] +', '
    f.write(prntStr.encode('utf8')+'\n')

# make a plot showing what fraction of all posts are made by users with the 
# label "female" across all categories
femGroupSum=twtWlabledRTs.loc[twtWlabledRTs.Gender=='female','RT_count'].groupby(
    twtWlabledRTs.Labels1).sum()/twtWlabledRTs.loc[:,'RT_count'].groupby(
    twtWlabledRTs.Labels1).sum()
femGroupSum.plot(kind='line')
plt.xlabel('Topic number')
plt.ylabel('Post proportion')
plt.title('Proportion of all tweets by females per category')
plt.show()

# make a plot showing what fraction of all posts are made by users with the 
# label "male" across all categories
maleGroupSum=twtWlabledRTs.loc[twtWlabledRTs.Gender=='male','RT_count'].groupby(
    twtWlabledRTs.Labels1).sum()/twtWlabledRTs.loc[:,'RT_count'].groupby(
    twtWlabledRTs.Labels1).sum()
maleGroupSum.plot(kind='line')
plt.xlabel('Topic number')
plt.ylabel('Post proportion')
plt.title('Proportion of all tweets by females per category')
plt.show()

# make a plot showing the difference between the fraction of male and female
# posters. Low values indicate categories that skew female, while high values
# indicate topics that skew male.

maleFemDelta=maleGroupSum-femGroupSum
maleFemDelta.plot(kind='line')
plt.xlabel('Topic number')
plt.ylabel('male-female proportion difference')
plt.title('male proportion-female proportion per category')
plt.show()

# Group data into 1 month chunks-It seems appropriate for the dataset
dateGroupsYrMo=twtWlabledRTs.loc[:,'Source'].groupby(
	[twtWlabledRTs.Date.dt.year,twtWlabledRTs.Date.dt.month]).count()

# show bar graphs of tweet proportion between topics over all date groups.
col=[]
combDF=pd.DataFrame()
for yrMoTup in dateGroupsYrMo.index:
    yrMoBool=((twtWlabledRTs.Date.dt.year==yrMoTup[0]).values &
              (twtWlabledRTs.Date.dt.month==yrMoTup[1]).values)
    totPosts=twtWlabledRTs.loc[yrMoBool,'RT_count'].sum()
    groupByLabels=twtWlabledRTs.loc[yrMoBool,'RT_count'].groupby(
        twtWlabledRTs.Labels1).sum()/totPosts
    groupByLabels.plot(kind='bar')
    plt.xlabel('Topic #')
    plt.ylabel('Number of Posts')
    plt.title(
    	'Posts in each category, incl. RTs for (year,month) ' + str(yrMoTup))
    plt.show()
    groupByLabels=groupByLabels.reset_index()
    combDF=pd.concat([combDF,groupByLabels.RT_count],axis=1,ignore_index=False)
    col.append(str(yrMoTup))

# combDFtrans has topic labels in columns position, with the monthly groupings
# along the row axis. The values are the proportion of:
# (Tweets in topic for month)/(total tweets in month)
combDF.columns=col
combDFtrans=combDF.T

topCategories=twtWlabledRTs.loc[:,'RT_count'].groupby(
    twtWlabledRTs.Labels1).sum()
topCategories.sort(ascending=False)

# plot specific labels (top 3)
combDFtrans[topCategories[:3].index].plot(kind='line')
plt.xlabel('Date')
plt.ylabel('proportion of posts')
plt.title('proportion of posts over time, top 3 categories')