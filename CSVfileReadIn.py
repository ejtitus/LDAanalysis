import pandas as pd
import os as os

def csvReadIn(filestring):
  header_names=['Date','Source','Handle1','Handle2','Gender',
          'Content']
  return(pd.read_csv(filestring,infer_datetime_format=True,
          parse_dates=[0],names=header_names,
          skiprows=[0,1],encoding='utf-8',
          dtype={'Content': object},
          na_values={'Content',''}))

def ReadInAndCombine():
	pname=''
	fileList=['Data1.csv']
	dfList=[]
	for f in fileList:
		dfList.append(csvReadIn(os.path.join(pname,f)))
	data=pd.concat(dfList,ignore_index=True)
	del(dfList)
	data.dropna(inplace=True)
	return(data)

def IfList(x):
	if len(x)>0:
		return(x[0])
	else:
		return(u'')

def ExtractRT(dataSeries):
	temp=dataSeries.str.findall((u'(?:RT) (@\w+)'))
	return(temp.apply(IfList))

def FormattedContentCol(dataSeries):
	return(dataSeries.str.replace((u'(?:RT) (@\w+:\s)'),'').str.lower())

def DropShort(df):
	df['WordCount']=df.Content.apply(lambda x: len(x.split()))
	df.drop(df[df.WordCount<3].index,inplace=True)
	return(df)

def DropRT(df):
	return(df[df.RT.str.len()<1])

def TweetsOnly(df):
	return(df[df.Source=='twitter'])

