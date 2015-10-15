LDAclustering README, Eric Titus, October 2, 2015

To use LDA via Gensim/python to cluster the text data contained in the Data1 csv file, run MainProgram.py, followed by AnalysisProgram.py

Currently, the code in RunLDAgensim groups the data into 35 topics, as determined by the code in OptimizeTopicNumber.py

This code relies on the following external packages:

numpy
scipy
pandas      "pip install pandas" at the command line will install (on linux)
nltk         natural language toolkit-need the most recent version "pip install -U nltk"
matplotlib   "pip install matplotlib" used for plotting
seaborn      "pip install seaborn" Seaborn makes the plots look better
gensim       "pip install gensim" This is the package that does the LDA


Module description:
CSVfileReadIn.py:    handles reading in of csv files to dataframes
UfileToList.py:      handles reading in of unicode file to a list
Tokenizer.py:        handles tokenization of social media posts
PrepareCorpusLDA.py: makes dictionary and corpus for LDA analysis
RunLDAgensim.py:     starts the LDA run


Main scripts:
MainProgram.py:         reads in csv and runs LDA analysis
AnalysisProgram.py:     uses LDA results in data analysis of posts
OptimizeTopicNumber.py: runs LDA analysis on a fraction of the data, looks for optimal topic #
