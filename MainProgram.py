# MainProgram.py,  Eric Titus,  October 1, 2015
# Read in CSV file containing ['Date','Source','Handle1',
				  # 'Handle2','Gender','Content']
# into a Pandas dataframe, then output filtered 'Content'
# data into a text file for clustering, and pickle the dataframe
# for later. Next, run clustering and analysis

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


# This code makes a word-> vector dictionary and
# a TF-IDF corpus of the text datafile generated
# in the tokenization step. These are saved to file
# done using Gensim

from PrepareCorpusLDA import MakeDictAndCorpus

filename='TweetContent.txt'
baseFilename=re.split('[.]',filename)[0]

MakeDictAndCorpus('TweetContent.txt') # This may take ~20 minutes



from RunLDAgensim import RunLDAinGensim #contains code to run LDA

# Run The full LDA, taking the input filenames for the dict/corpus/output file
dictFile=baseFilename + '.dict'
corpusFile=baseFilename+'TFIDF.mm'
outputFile=baseFilename + '.LDA'

RunLDAinGensim(dictFile,corpusFile,outputFile)

#reload the output file
from gensim import models,corpora

lda=models.ldamodel.LdaModel.load(baseFilename + '.LDA')
mmTFIDF = corpora.MmCorpus(baseFilename+'TFIDF.mm')
dictionary=corpora.Dictionary.load(baseFilename + '.dict')

import pyLDAvis.gensim
data_created_using_prepare = pyLDAvis.gensim.prepare(lda, mmTFIDF, dictionary)
pyLDAvis.save_html(data_created_using_prepare, 'LDAvis.html')

#Run analysis program next to analyze results