#!/usr/bin/env python
# coding: utf-8

# In[37]:


from pymongo import MongoClient


# In[38]:



client = MongoClient('mongodb+srv://AmazonianSentiments:6pVOMaDeacyVgrre@amazoniansentiments.duy3v.mongodb.net/AmazonianSentiments?retryWrites=true&w=majority')
mydb = client["AmazonianSentiments"] #pyramids is the database
mycol = mydb["AllBeauty"] #invoice is the collection


# In[39]:


import nltk
import pandas as pd 
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from pymongo import MongoClient
from pandas.io.json import json_normalize


# In[40]:


cursor=mycol.find()
df = pd.json_normalize(cursor)


# In[41]:


df


# In[44]:


# replace field that's entirely space (or empty) with NaN
df.replace(r'^\s*$', np.nan, regex=True)
nltk.download('stopwords')
#Create column of review text with all lowercase, no punctuation, and no stopwords
nan_value = float("NaN") #Create na variable for blanks
df["reviewText"].replace("", nan_value, inplace=True) #Replace blanks with na variable
df.dropna(subset = ["reviewText"], inplace=True) #Drop all rows with na review text
df["ReviewNoFiller"] = df["reviewText"].str.replace('[^\w\s]','') #Create column with review text with no punctuation
df["ReviewNoFiller"] = df["ReviewNoFiller"].str.lower() #Make all words lowercase
stopwords = stopwords.words('english')
df["ReviewNoFiller"] = df["ReviewNoFiller"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) #Remove stop words
df["ReviewNoFiller"].replace("", nan_value, inplace=True) #Replace blanks with na
df.dropna(subset = ["ReviewNoFiller"], inplace=True) #Drop all rows with na review text


# In[46]:


#Create column of summary text with all lowercase, no punctuation, and no stopwords
df["SummaryNoFiller"] = df["summary"].str.replace('[^\w\s]','') #Create column with summary text with no punctuation
df["SummaryNoFiller"] = df["SummaryNoFiller"].str.lower() #Make column all lowercase
df["SummaryNoFiller"] = df["SummaryNoFiller"].fillna("") # Remove NA values
df["SummaryNoFiller"] = df["SummaryNoFiller"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))


# In[47]:


#Insert columns with tokenized review and summary
df["ReviewToken"] =df.apply(lambda row: word_tokenize(row["ReviewNoFiller"]), axis=1)
df["SummaryToken"] = df.apply(lambda row: word_tokenize(row["SummaryNoFiller"]), axis=1)


# In[49]:


#Insert column with review word count
df["WordCount"] = df["ReviewToken"].apply(len)
#Add column with Date from converted Unix time, remove redundant columns
#df["Date"] = pd.to_datetime(PetSuppliesDF["unixReviewTime"], unit='s')
#df = df.drop(['reviewTime', 'unixReviewTime'], axis=1)


# In[51]:


#Add Column for Numbers of Reviews for Reviewer
ReviewCount = df.groupby('reviewerID').asin.nunique().to_frame()
ReviewCount.reset_index(inplace=True)
ReviewCount = ReviewCount.rename(columns = {'asin':'ReviewNum'})
df = df.merge(ReviewCount, on = 'reviewerID')


# In[52]:


#Add Column for Average Star Rating for Reviewer
ReviewAvg = df.groupby('reviewerID')['overall'].mean().to_frame()
ReviewAvg.reset_index(inplace=True)
ReviewAvg = ReviewAvg.rename(columns = {'overall':'AvgRating'})
df = df.merge(ReviewAvg, on = 'reviewerID')


# In[54]:


#Determine the gender in df
import gender_guesser.detector as gender
d = gender.Detector()
first_names = []
for i in range(0,370527):
    name = str(df['reviewerName'].values[i]).split(' ', 1)[0]
    first_names.append(name)
# lowercase everything
first_names = [k.lower() for k in first_names]
# capitalize the first letter of every name
first_names = [i.capitalize() for i in first_names]

genders = []
for i in first_names[0:len(first_names)]:
    if d.get_gender(i) == 'male':
        genders.append('male')
    elif d.get_gender(i) == 'female':
        genders.append('female')
    else:
        genders.append('unknown')


# In[55]:


df['genders'] = genders


# In[45]:


df


# In[ ]:


#Split dataframe into train (60%), validate (20%), and test (20%) dataframes with rows randomized
#train, validate, test = \
              #np.split(df.sample(frac=1, random_state=42), 
                       [int(.6*len(df)), int(.8*len(df))])

#Print some of the dataframe to verify work
#pd.set_option('display.max_columns', None) #So as not to truncate output
#pd.set_option('display.max_rows', None) #So as not to truncate output
#for col in df.columns: #Print column names
    #print(col)
#print(PetSuppliesDF.head(1)) # Print first entry in dataframe

# Write final dataframe into csv
#df.to_csv(r'PetSupplies.csv', index = False)


# In[56]:


df


# In[57]:


df['genders'].value_counts()

