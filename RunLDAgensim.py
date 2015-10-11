#RunLDAgensim.py,   Eric Titus,  October 1, 2015
# After running PrepareCorpusLDA.py, this code reads in the TFIDF
# weighted corpus and dictionary to apply LDA.

# Import all relevant packages
import logging, gensim
from gensim import corpora, models, similarities, matutils
from time import time
import codecs
import numpy as np

def RunLDAinGensim(dictFilename,corporaFilename,outputFilename,):

	#Read in dictionary and TFIDF weighted Corpus
	dictionary=corpora.Dictionary.load(dictFilename)
	mmTFIDF = gensim.corpora.MmCorpus(corporaFilename)

	lda = gensim.models.ldamodel.LdaModel(corpus=mmTFIDF, 
		id2word=dictionary, num_topics=36, update_every=1, 
		chunksize=10000, passes=1)

	lda.save(outputFilename)
