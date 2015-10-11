# OptimizeTopicNum.py,  Eric Titus,  October 1, 2015
# Determine optimal number of clusters for LDA analysis of Twitter data.
# You can use this code with a dictionary and corpus already made, 

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

# use if working in ipython notebook
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import seaborn as sns  #this makes the plots look good. 
sns.set_context("notebook", font_scale=1.5, 
	rc={"lines.linewidth": 2.5})


# set this to true is 
makeNewTokenizedText=True


if makeNewTokenizedText:

	# Import subfunction for csv read in output is pandas Dataframe
	# expected columns=['Date','Source','Handle1',
					  # 'Handle2','Gender','Content']
	from CSVfileReadIn import ReadInAndCombine, ExtractRT, DropShort, TweetsOnly, csvReadIn

	#Generate full pandas dataframe of all CSVfiles
	#FYI, the CSVfile names are hard-coded in "ReadInAndCombine"
	data=ReadInAndCombine()

	# use this line for a unique filename
	# data=csvReadIn(filename)

	data['RT']=ExtractRT(data.Content)  #put RTs into new column
	data=DropShort(data)  #cut out tweets of less than 3 words

	dataNoRT=data[data.RT.str.len()<1]  #make a frame with no RTs
	dataRTonly=data[data.RT.str.len()>1] #make a frame with only RTs
	dataTweets=TweetsOnly(data).copy()

	# These lines format the tweet content and add it to the dataframe.
	# The formatting helps match up RTs, so that we can count them up.

	dataTweets.loc[:,'Content_Formatted']=dataTweets.loc[:,('Content')]

	from UfileToList import UfileToList #UfileToList imports unicode text into a list
	stopwords=UfileToList('stopwordList.txt') # loads in stopwords file

	#applies tokenization and cleanup steps,
	#then writes to text file (takes input filename)
	from Tokenizer import fullTokDict, FormatContent

	# formats tweets like they would be to be used in NLP. This helps weed out
	# duplicates. If I had more time, I would make this better.
	dataTweets.loc[:,'Content_Formatted']=FormatContent(
										dataTweets.loc[:,'Content_Formatted'],
										stopwords)

	# Make a dataframe which has formatted content in one col, # of RTs in the other
	contRepS=dataTweets.groupby(['Content_Formatted']).size()
	contentRepeats=contRepS.reset_index()
	contentRepeats.columns=['Content_Formatted','RT_count']

	# in main frame, remove duplicate tweets (RTs).
	dataTweetsNoDup=dataTweets.drop_duplicates('Content_Formatted')

	# Add back in a count of the repeated RTs
	twtWlabledRTs=pd.merge(
		dataTweetsNoDup,contentRepeats, on='Content_Formatted', how='left')

	# take tweet content (without duplicates) and tokenize/format it,
	# then save each tweet to a line in 'filename'

	filename='TweetContent.txt'

	fullTokDict(
		twtWlabledRTs.Content.str.lower().values,stopwords,
		filename)

	# Pickle all the pandas dataframes for later use
	f=open('ProcessedDataFrame.pckl','wb')
	pickle.dump(data,f,protocol=-1)
	f.close()

	dataNoRT=data[data.RT.str.len()<1]
	f=open('DataFrameNoRT.pckl','wb')
	pickle.dump(dataNoRT,f,protocol=-1)
	f.close()

	dataRTonly=data[data.RT.str.len()>1]
	f=open('DataFrameRTonly.pckl','wb')
	pickle.dump(dataRTonly,f,protocol=-1)
	f.close()

	f=open('twtWlabledRTs.pckl','wb')
	pickle.dump(twtWlabledRTs,f,protocol=-1)
	f.close()

# To unpickle saved pandas DF:

# f=open('twtWlabledRTs.pckl','rb')
# twtWlabledRTs=pickle.load(f)
# f.close()

# cut down corpus to a manageble size. Anything more than about 50k posts gives
# a memory error when calculating the error in the modeling

filename='TweetContent.txt'
baseFilename=re.split('[.]',filename)[0]

maxCorpusLen=50000
# maxCorpusLen=5000

corpusLen=0
f = codecs.open(filename, encoding='utf-8')
for line in f:
	corpusLen+=1
f.close()

stepsize=corpusLen/maxCorpusLen

smallTokDataFile=baseFilename + 'Sub.txt'
baseFilename=re.split('[.]',smallTokDataFile)[0]

f=codecs.open(filename,encoding='utf-8')
fOut=open(smallTokDataFile,'w')
counter=1
for line in f:
    if counter % stepsize ==0:
        fOut.write(line.encode('utf8'))
        counter+=1
    else:
        counter+=1
f.close()
fOut.close()


# This code makes a word-> vector dictionary and
# a TF-IDF corpus of the text datafile generated
# in the tokenization step. These are saved to file
# done using Gensim

from PrepareCorpusLDA import MakeDictAndCorpus
from gensim import corpora, models, similarities, matutils
import scipy.stats as stats

# filename='TweetContent.txt'
# baseFilename=re.split('[.]',filename)[0]

MakeDictAndCorpus(smallTokDataFile) # This may take ~20 minutes

# reload the dictionary and corpus
dictFile=baseFilename + '.dict'
corpusFile=baseFilename+'TFIDF.mm'
mmTFIDF = corpora.MmCorpus(baseFilename+'TFIDF.mm')
dictionary=corpora.Dictionary.load(baseFilename + '.dict')

# These functions give something called the KL divergence. This will be 
# minimized when the proper number of topics is achieved
# see http://blog.cigrainger.com/2014/07/lda-number.html
def sym_kl(p,q):
    return np.sum([stats.entropy(p,q),stats.entropy(q,p)])

l = np.array([sum(cnt for _, cnt in doc) for doc in mmTFIDF])
def arun(my_corpus,dictionary,min_topics,max_topics,step):
    kl = []
    for i in range(min_topics,max_topics,step):
        t1=time()
        lda = models.ldamodel.LdaModel(corpus=my_corpus,
            id2word=dictionary,num_topics=i,update_every=1,passes=5)
        m1 = lda.expElogbeta
        U,cm1,V = np.linalg.svd(m1)
        #Document-topic matrix
        lda_topics = lda[my_corpus]
        m2 = matutils.corpus2dense(lda_topics, lda.num_topics).transpose()
        cm2 = l.dot(m2)
        cm2 = cm2 + 0.0001
        cm2norm = np.linalg.norm(l)
        cm2 = cm2/cm2norm
        kl.append(sym_kl(cm1,cm2))
        print(time()-t1)
    return kl

# set range of topics to explore for LDA analysis
minTopics=2
maxTopics=200
stepSize=2

kl = arun(mmTFIDF,dictionary,minTopics,maxTopics,stepSize)

# Plot kl divergence against number of topics. The minimum 
plt.plot(range(minTopics,maxTopics,stepSize),kl)
plt.ylabel('Symmetric KL Divergence')
plt.xlabel('Number of Topics')
plt.savefig('kldiv.png', bbox_inches='tight')