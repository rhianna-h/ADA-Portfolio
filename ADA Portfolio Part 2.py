#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv('portfolio.csv')
df.head()


# In[6]:


df = df.set_axis(["ID", "Title", "Comments",
              "Journal Reference", "Categories"], axis="columns")


# In[7]:


df = df[df["Title"].str.contains("covid|covid19|covid-19|COVID|coronavirus")
         == False]
#df


# In[8]:


df_new= df.dropna(subset=['Comments'])
df_new.head()


# In[9]:


categories = df_new['Categories'].str.split(' ', expand=True)
categories.head()


# In[10]:


original = ['cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 
                'cs.CL', 'cs.CR', 'cs.CV', 'cs.CY', 'cs.DB',
               'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET',
                'cs.FL', 'cs.GL', 'cs.GR', 'cs.GT', 'cs.HC',
                'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO', 'cs.MA',
                'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE', 'cs.NI',
                'cs.OH', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO',
               'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY',
                
               'econ.EM', 'econ.GN', 'econ.TH',
                
                'eess.AS', 'eess.IV', 'eess.SP', 'eess.SY',
                
                'math.AC', 'math.AG','math.AP', 'math.AT', 'math.CA',
                'math.CO', 'math.CT', 'math.CV','math.DG', 'math.DS',
                'math.FA', 'math.GM', 'math.GN', 'math.GR','math.GT',
                'math.HO', 'math.IT','math.KT', 'math.LO', 'math.MG',
               'math.MP', 'math.NA', 'math.NT','math.OA', 'math.OC',
                'math.PR','math.QA', 'math.RA', 'math.RT','math.SG', 
                'math.SP', 'math.ST',
                
                'astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA', 'astro-ph.HE', 'astro-ph.IM',
                'astro-ph.SR', 'cond-mat.dis-nn', 'cond-mat.mes-hall', 'cond-mat.mtrl-sci', 'cond-mat.other',
                'cond-mat.quant-gas', 'cond-mat.soft', 'cond-mat.stat-mech', 'cond-mat.str-el', 'cond-mat.supr-con',
                'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph', 'hep-th',
                'math-ph', 'nlin.AO', 'nlin.CD', 'nlin.CG', 'nlin.PS',
                'nlin.SI', 'nucl-ex', 'nucl-th', 'physics.acc-ph', 'physics.ao-ph',
                'physics.app-ph', 'physics.atm-clus', 'physics.atom-ph', 'physics.bio-ph', 'physics.chem-ph',
                'physics.class-ph', 'physics.comp-ph', 'physics.data-an', 'physics.ed-ph', 'physics.flu-dyn',
                'physics.gen-ph', 'physics.geo-ph', 'physics.hist-ph', 'physics.ins-det', 'physics.med-ph',
                'physics.optics', 'physics.plasm-ph', 'physics.pop-ph', 'physics.soc-ph', 'physics.space-ph',
                'quant-ph', 'supr-con', 'cond-mat','chao-dyn','astro-ph',
           'acc-phys', 'plasm-ph', 'atom-ph', 'chem-ph', 'mtrl-th', 
           'comp-gas', 'q-alg', 'funct-an', 'ao-sci', 'cmp-lg',
           'alg-geom', 'solv-int', 'dg-ga', 'patt-sol', 'adap-org',
           'bayes-an',
           
           
                
                'q-bio','q-bio.PE','q-bio.TO', 'q-bio.BM', 'q-bio.CB', 
                'q-bio.GN', 'q-bio.MN', 'q-bio.NC','q-bio.OT','q-bio.PE',
                'q-bio.QM', 'q-bio.SC',
                
                'q-fin.CP', 'q-fin.EC','q-fin.TR','q-fin.GN', 'q-fin.MF', 
                'q-fin.PM','q-fin.PR', 'q-fin.RM', 'q-fin.ST',
               
                'stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.OT', 
                'stat.TH'
               ]      


# In[11]:


newcategories = ['Computer Science', 'Computer Science', 'Computer Science','Computer Science','Computer Science',
                'Computer Science','Computer Science','Computer Science','Computer Science','Computer Science',
                'Computer Science','Computer Science','Computer Science','Computer Science','Computer Science',
                'Computer Science','Computer Science','Computer Science','Computer Science','Computer Science',
                'Computer Science','Computer Science','Computer Science','Computer Science','Computer Science',
                'Computer Science','Computer Science','Computer Science','Computer Science','Computer Science',
                 'Computer Science','Computer Science','Computer Science','Computer Science','Computer Science',
                'Computer Science','Computer Science','Computer Science','Computer Science','Computer Science',
                 
                 'Economics', 'Economics', 'Economics', 
                 
                 'Electric', 'Electric', 'Electric', 'Electric', 
                 
                 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 
                 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics',
                 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics',
                 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics',
                 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics',
                 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics',
                 'Mathematics', 'Mathematics',
                 
                 'Physics', 'Physics', 'Physics','Physics','Physics',
                 'Physics', 'Physics', 'Physics','Physics','Physics',
                 'Physics', 'Physics', 'Physics','Physics','Physics',
                 'Physics', 'Physics', 'Physics','Physics','Physics',
                 'Physics', 'Physics', 'Physics','Physics','Physics',
                 'Physics', 'Physics', 'Physics','Physics','Physics',
                 'Physics', 'Physics', 'Physics','Physics','Physics',
                 'Physics', 'Physics', 'Physics','Physics','Physics',
                 'Physics', 'Physics', 'Physics','Physics','Physics',
                 'Physics', 'Physics', 'Physics','Physics','Physics',
                 'Physics','Physics', 'Physics','Physics', 'Physics',
           'Physics', 'Physics', 'Physics','Physics', 'Physics',
           'Physics','Physics', 'Physics', 'Physics', 'Physics',
           'Physics','Physics', 'Physics', 'Physics', 'Physics',
           'Physics',
                 
                 'Quantitative Biology', 'Quantitative Biology', 'Quantitative Biology', 'Quantitative Biology','Quantitative Biology',
                 'Quantitative Biology', 'Quantitative Biology', 'Quantitative Biology','Quantitative Biology','Quantitative Biology',
                 'Quantitative Biology','Quantitative Biology',
                 
                 'Quantitative Finance', 'Quantitative Finance','Quantitative Finance', 'Quantitative Finance', 'Quantitative Finance', 
                 'Quantitative Finance','Quantitative Finance', 'Quantitative Finance', 'Quantitative Finance',
                 
                 'Statistics', 'Statistics', 'Statistics', 'Statistics', 'Statistics', 
                 'Statistics'
        
                ]


# In[12]:


categoriesnew1=categories.replace(original, newcategories)


# In[13]:


col = categoriesnew1.iloc[:,[0]]


# In[14]:


df_new.insert(5,'Categories New', col )


# In[15]:


df_new.iloc[:, 5]


# In[83]:


from sklearn.preprocessing import LabelEncoder as le


# In[86]:


from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
df_new["Categories Coded"] = ord_enc.fit_transform(df_new[["Categories New"]])
df_new[["Categories New", "Categories Coded"]].head(11)


# In[93]:


ord_enc = OrdinalEncoder()
df_new["Title Coded"] = ord_enc.fit_transform(df_new[["Title"]])


# In[94]:


df_new.head()


# In[95]:


df_new.iloc[:, 7]


# In[38]:


#df_new['Categories New'] = le.fit_transform(df_new['Categories New'])


# In[ ]:





# In[96]:


X = df_new.iloc[:, 6].values.reshape(-1, 1)  # values converts it into a numpy array
Y = df_new.iloc[:, 7].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions


# In[100]:


print(f"intercept: {linear_regressor.intercept_}")


print(f"slope: {linear_regressor.coef_}")


# In[97]:


plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[98]:


Y_pred


# ### Title & Recommendations

# In[17]:


title_df = df_new.iloc[:, [1]]
#titledf
length = df_new['Title'].str.len()
words = df_new['Title'].str.split().str.len()


# In[18]:


title_df['Title Length'] = length
title_df['Total Words'] = words


# In[19]:


title_df.head()


# In[27]:




mean_length = title_df['Title Length'].mean()
mode_length = title_df['Title Length'].mode()
median_length = title_df['Title Length'].median()

print('The mean of the titles length is ', mean_length)
print('The mode of the titles length is ', mode_length)
print('The media of the ttiles length is ', median_length)


# In[26]:


mean = title_df['Total Words'].mean()
print("The title word mode is", mean)
mode = title_df['Title Length'].mode()
print("The title length mode is", mode)

mean = title_df['Total Words'].mean()
print("The title word mode is", mean)


# In[7]:


import re
from collections import Counter
from collections import Counter
import string
import nltk


# In[40]:


titles = str.join(",", title_df['Title'])
words = re.sub("/n|[^a-zA-Z]","", titles)
words = re.sub(" +", " ", words)
words = str.lower(words)
words = words.split(" ")
Counter = Counter(words)
frequency = Counter.most_common()


# In[9]:


titlecommon=df_new['Title'].str.lower()


# In[10]:


titlecommon = titlecommon.values.tolist()


# In[12]:


textbreak = [nltk.tokenize.wordpunct_tokenize(text) for text in titlecommon]


# In[13]:


stopwords = nltk.corpus.stopwords.words('english')


# In[14]:


from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
punctuation = string.punctuation # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# Add numbers
punctuation += '0123456789'

def comment_raiz(comment):
    text = []
    for lista in comment:
        valids = [stemmer.stem(word) for word in lista if word not in stopwords and word not in punctuation 
                  and len(word)>2]
        valids_true = [''.join([char for char in word if char not in punctuation]) for word in valids if 
                       len(''.join([char for char in word if char not in punctuation]))>0]
        text.append(valids_true)
    return text


# In[15]:


titleclear = comment_raiz(textbreak)


# In[16]:


def counter(comment_clear):
    cnt = Counter()
    for words in comment_clear:
        for word in words:
            cnt[word] += 1
    return cnt


# In[17]:


titlecommon = counter(titleclear)


# In[18]:


top10 =titlecommon.most_common(10)


# In[19]:


top10


# In[21]:


import seaborn as sns
f = pd.DataFrame(top10)
f = f.rename(columns = {0: "Word", 1: "Count"})

fig, ax = plt.subplots()
sns.kdeplot(f["Count"])
plt.show() #slide 33


# In[30]:


import timeit


# In[38]:


s = title_df['Title'].replace(' ', ',')
s


# In[47]:


s = df_new['Title'].replace(" ", "")


# In[48]:


s


# In[41]:


title_df['Title Words'] = title_df['Title'].str.strip()


# In[42]:


title_df


# In[37]:


title_df


# start = timeit.default_timer()
# 
# def get_weight(title_words):
#     weight = 0
#     tw = title_words.split(',')
#     for word in tw:
#         if (word in stopwords): pass
#         elif (word in class1): weight = weight + 20
#         elif (word in class2): weight = weight +5
#         else: weight=weight=1
#     return weight
# 
# title_df["Weight"] = title_df["Total Words"].apply(lambda tw: get_weight(tw))
# stop = timetit.default_timer()
# 
# print("time: ", stop-start)
# t.head()

# ### Recommendations

# In[50]:


top10


# In[54]:


has_jref = df_new.loc[~df_new["Journal Reference"].isna()]
has_jref.head()


# In[58]:


def find_top_word(title):
    title = title.lower()
    title = title.split(" ")
    
    for keyw in top10:
        if keyw[0] in title:
            return keyw[0]
        
top_ten_art = has_jref
top_ten_art["Top 10 Keyword"] =  top_ten_art["Title"].apply(lambda t: find_top_word(t))
top_ten_art = top_ten_art.loc[~top_ten_art["Top 10 Keyword"].isna()]
top_ten_art.head()


# In[28]:


data0 = [['1', 1721516], ['2', 774895], ['3', 281718],
       ['4', 87838], ['5', 22860], ['6', 4594],
    ['7', 727], ['8', 158], ['9', 43], 
    ['10', 17], ['11', 3],
    ['12', 1], ['13', 1]
       ]
  
# Create the pandas DataFrame
amountpercat = pd.DataFrame(data0, columns=['Category', 'Articles Amount'])
  


# In[37]:


hist = amountpercat.hist()


# In[ ]:




