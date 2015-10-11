#PrepareCorpusLDA.py,   Eric Titus,  October 1, 2015
# After running ImportAndGenerateCorpus.py, this code prepares
# the corpus (TokenizedData.txt) for analysis in Gensim using
# TFIDF term weighting followed by Latent Dirichlet Allocation (LDA)

# Import all relevant packages
import logging, gensim
from gensim import corpora, models
from time import time
import codecs
import re
import os

def MakeDictAndCorpus(filename):

	baseFilename=re.split('[.]',filename)[0]

	# Open the file containing the text corpora
	f = codecs.open(filename,encoding='utf-8') 

	# Read in Text corpora to dictionary: creates hash for words to matrix location
	dictionary = corpora.Dictionary([[word for word in line.split()] for line in f])
	f.close()

	dictionary.filter_extremes(no_below=5, no_above=0.5)


	dictionary.save(baseFilename + '.dict') # save for later

	# Define iterable object to contain text corpus. This allows 
	# for streaming of text data from a file, giving memory independence

	class MyCorpus(object):
	    def __iter__(self):
	        for line in codecs.open(filename,encoding='utf-8'):
	            yield dictionary.doc2bow(line.split())

	corpus = MyCorpus()

	tfidf = models.TfidfModel(corpus) #wrap corpus in TFIDF model

	# Save serialized TFIDF corpus to disk, to be run later with gensim.
	corpora.MmCorpus.serialize(baseFilename+'TFIDF.mm', tfidf[corpus])

