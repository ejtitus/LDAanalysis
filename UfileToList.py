
import codecs
def UfileToList(filename):
	f = codecs.open(filename,
   					encoding='utf-8')
	stoplist=[];

	for line in f:
		stoplist.append(unicode(line)[:-1])
	f.close()
	return(stoplist)