# Tokenizer.py,  Eric Titus, Sept. 30, 2015
# contains functions used to tokenize social media posts
# Writes tokenized strings to a file for later reading (TokenizedData.txt)

# need built in tokenizers from NLTK. May need to be updated
from nltk.tokenize import TweetTokenizer,word_tokenize, sent_tokenize
import re
import pandas as pd

# tknzrTweet takes in a multi-word string, and spits out a list of single words
# uses the NLTK TweetTokenizer, which has some good Tweet specific features
def tknzrTweet(txt):
    tknzr=TweetTokenizer(strip_handles=True, reduce_len=True)
    return(tknzr.tokenize(txt))

# tknzrStd still needs work- i.e. it doesn't like tweets or unicode, I didn't 
# try to figure out which
def tknzrStd(txt):
# '''Tokenizes into sentences, then strips punctuation/abbr, converts to lowercase and tokenizes words'''
    return [word_tokenize(" ".join(re.findall(u'\w+', t,
                        flags = re.UNICODE | re.LOCALE)).lower()) 
            for t in sent_tokenize(txt.replace("'", ""))]


# FullTokDict takes a list of sentence strings, tokenizes them, strips stopwords
# and strips other symbols, then outputs the cleaned strings to a text file,
# where each tweet/piece of content is a line (words split with spaces)
def fullTokDict(documents,stopwords,filename):
    
    f=open(filename,'w') #open output file
    
    for doc in documents:  #documents should be iterable, with 1 sentence/line
        tempDoc=tknzrTweet(doc) #splits sentence into tokens(words)
        tempDoc2='' # keep good words in this string
        for word in tempDoc:  # iterate over tokens from tknzrTweet
            if word not in stopwords:  #stopword removal

                # This is an ugly regex line, but it works-added to it as needed
                tempDoc2+=(re.sub(
                    u'^https?:\/\/.*[\r\n]*|^\w\s|\d|^\W\s|\W{2,10}|'
                    ':.+|^\w\s|@\w+:','',word)+u' ')
#         tempDoc2=filter(None,tempDoc2)
        f.write((tempDoc2+u'\n').encode('utf8')) # write the output line to f
    f.close()

# FormatContent is similar to fullTokDict, but is used to format a column of 
# a pandas dataframe. I do this so that we can eliminate duplicate posts before
# using fullTokDict to output the data to a file.
def FormatContent(dataSeries,stopwords):
    documents=dataSeries.str.lower().values #pull content text from series
    docOut=[];
    for doc in documents: # again, iterate over each tweet/post
        tempDoc=tknzrTweet(doc) # tokenize with the twitter specific tokenizer.
        tempDoc2=''
        for word in tempDoc:  #loop over each word in the line...
            if word not in stopwords: #pull stopwords.
                tempDoc2+=(re.sub(
                    u'^https?:\/\/.*[\r\n]*|^\w\s|\d|^\W\s|\W{2,10}|'
                    ':.+|^\w\s|@\w+:|(?:RT) (@\w+:\s)','',word)+u' ')
#         tempDoc2=filter(None,tempDoc2)
        tempDoc2=re.sub(' +',' ',tempDoc2)
        docOut.append(tempDoc2) #append value to list so that it can be returned
    return(pd.Series(docOut,dataSeries.index))