#!/usr/bin/env python
# coding: utf-8

# In[2]:


import dask.dataframe as dd
import pandas as pd
import numpy as np


# In[4]:


import seaborn as sns
import plotly as px
import matplotlib.pyplot as plt


# In[18]:


ddf= dd.read_json('arxiv_v2.json', blocksize = '16MB',  #Read the json file using the following
                 sample = 2**26, encoding = 'utf-8',
                 dtype ={'id': 'object'})
ddf # have a laxy output of the dataframe


# In[19]:


ddf.head(5)


# In[20]:


ddf.columns #Gain an understanding of what the columns are called


# In[21]:


ddf.npartitions #The number of partitions


# In[22]:


ddf.dtypes


# In[23]:


ddf.drop_duplicates()  


# In[24]:


mem = ddf.memory_usage(index = True, deep = True).compute()
mem


# In[15]:


import math

def convert_size(size_bytes):
    label = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    nth_1 = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, nth_1)
    size = round(size_bytes / p, 2)
    return "%s %s" % (size, label[nth_1])

convert_size(mem.sum()) # working out the size of the file


# In[26]:


print('The size of the original file is ', convert_size(mem.sum()) )


# In[27]:


shape = ddf.shape
print("Number of rows: ", shape[0].compute())
print("Number of columns: ", shape[1])


# ### With the 5 columns

# In[28]:


ddf2 = ddf[['id', 'title', 'comments', 'journal-ref', 'categories']]
ddf2.head()


# In[29]:


ddf2.describe().compute()


# ### CSV file

# In[30]:


#ddf2.compute().to_csv('portfolio.csv', index=False)


# In[31]:


# Data Architecture Framework Techniques


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


df = pd.read_csv('portfolio.csv')
df.head()


# In[11]:


df = df.set_axis(["ID", "Title", "Comments",
              "Journal Reference", "Categories"], axis="columns")
df.head(1)


# In[12]:


df.info()


# In[13]:


mem_new = df.memory_usage(index = True, deep = True)
mem_new


# In[14]:


convert_size(mem_new.sum())


# In[16]:


shape_new = df.shape
print("Number of rows: ", shape_new[0])
print("Number of columns: ", shape_new[1])


# #### Mini Objective: Dataset with records containing comments and without the word COVID in the title. - complete

# ##### Data Wrangling

# In[17]:


covid1 = len(df[df["Title"].str.contains("covid|covid19|covid-19|COVID|coronavirus")]) 

#Calculate how many word contains coivd or its variants in the dataframe 

print('There are ', covid1, 'titles containing the word covid or similar')


# In[18]:


df = df[df["Title"].str.contains("covid|covid19|covid-19|COVID|coronavirus")
         == False]
#df


# In[19]:


nan_count = df.isna().sum() #How many null values are ther
print(nan_count )


# In[20]:


df['Comments'].isna().sum() #working out the total amount of null values


# In[21]:


# Drop only those rows where the specified column has a missing value
df_new= df.dropna(subset=['Comments'])
df_new.head()


# In[22]:


shape_new2 = df_new.shape
print("Number of rows: ", shape_new2[0]) #undersstanding how the dataset looks with columns removed
print("Number of columns: ", shape_new2[1])


# This is the cleaned version of the dataset, where it has 1721542 rows and 5 columns. This means the number of rows has decreased by 528681 and 9 columns removed.
# 
# The dataset no longer contains null values in the comments column, it has no duplicates and in the title column there is no mention of covid.

# ### Relationship between variables 
# Answering what happened question. Please provide appropriate visualisations according to the data you are describing. 

# In[39]:


df_new.head()


# ### Catgeories

# In[44]:


categoriesnew = pd.read_csv('categories_descending.csv')
categoriesnew.head() #From arxiv the data acroynyms for the categories are defined


# In[40]:


categories = df_new['Categories'].str.split(' ', expand=True)
categories.head()


# In[41]:


count =pd.DataFrame(categories.count())
count.rename(columns={
    count.columns[0]: "Categories Amount",
},inplace=True)
count["Number"] = count.index
count


# In[42]:


### Visualise the data count 


# In[44]:


data0 = [['1', 1721516], ['2', 774895], ['3', 281718],
       ['4', 87838], ['5', 22860], ['6', 4594],
    ['7', 727], ['8', 158], ['9', 43], 
    ['10', 17], ['11', 3],
    ['12', 1], ['13', 1]
       ]
  
# Create the pandas DataFrame
amountpercat = pd.DataFrame(data0, columns=['Category', 'Articles Amount'])
  


# In[46]:


fig = px.histogram(amountpercat, x="Category", nbins=20, title='Histogram showing the length of the title')
fig.show()


# In[32]:


import plotly.express as px
data = count

x=count['Number']
y=count['Categories Amount']

fig = px.bar(data, x=x, y=y, title='Amount per Category')
fig.show()


# In[49]:


null = categories.isnull().sum(1)
null


# ### Naming conventions and Replacing them

# In[33]:


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


# In[34]:


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


# In[35]:


#categoriesnew1 = categories.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]] #having specific columns from the categroies data
categoriesnew1=categories.replace(original, newcategories)
categoriesnew1


# In[36]:


import seaborn as sns


# In[37]:


u = categoriesnew1.nunique(axis=1)
fig, ax = plt.subplots(figsize=(12, 8))
sns.kdeplot(u)
plt.show() #creating a KDE plot to see the distribution


# In[38]:


col = categoriesnew1.iloc[:,[0]] #we only want to focus on one bit by merge and joining the other columnc
col


# In[59]:


col.value_counts()#count how many is in each category


# In[3]:


data1 = [['Physics', 1090930], ['Mathematics', 309706], ['Computer Science', 253426],
       ['Electric', 21848], ['Statistics', 21608], ['Quantitative Biology', 16288],
    ['Quantitative Finance', 5851], ['Economics', 1859]
       ]
  
# Create the pandas DataFrame
breakdown_of_cats = pd.DataFrame(data1, columns=['Category', 'Articles Amount'])
  
# print dataframe.
breakdown_of_cats


# In[4]:


import plotly.express as px

fig = px.scatter(breakdown_of_cats, x="Category", y="Articles Amount")
fig.show()


# In[5]:


fig = px.pie(breakdown_of_cats, values='Articles Amount', names='Category', 
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()


# ### New category column with all data

# In[60]:


df_new.insert(5,'Categories New', col ) #insert a new column which has the meaning of the Acronyms
df_new


# In[63]:


df_new.columns


# ### Journal Reference

# In[7]:


data2 = [['No Reference', 1041658], ['With Reference', 679858]]
journ = pd.DataFrame(data2, columns=['Journal Status', 'Value'])

journ


# In[8]:


fig = px.bar(journ, x='Journal Status', y='Value')
fig.show()


# In[ ]:





# In[93]:


ex = df_new[['Categories New', 'Journal Reference']]

cols = ['With Reference', 'Without Reference']
ex.loc[~ex['Journal Reference'].isnull(), cols] = [1,0]
ex.loc[ex['Journal Reference'].isnull(), cols] = [0,1]

ex.drop(columns = ['Journal Reference'], axis = 1)
ex = ex.groupby(['Categories New']).sum()
ex = ex.reset_index()
ex


# In[94]:


plt.rcParams.update({'font.size': 20})

ex.plot(
    x = 'Categories New',
    kind = 'barh',
    stacked = True,
    mark_right = True,
    title = 'Journal Reference by Category',
    figsize = (20,7)
)

plt.show()


# In[88]:


4130.0+17478.0


# In[89]:


4130/21608.0


# In[87]:


33490/253426.0


# In[ ]:





# In[73]:


ex


# In[91]:


plt.rcParams.update({'font.size': 20})

ex.plot(
    x = 'Categories New',
    kind = 'barh',
    stacked = True,
    mark_right = True,
    title = 'Journal Reference by Category',
    figsize = (20,7)
)

plt.show()


# In[77]:


import plotly.express as px

fig = px.pie(ex, values='With Reference PC', names='Categories New')
fig.show()


# ### Title

# In[101]:


title_df = df_new.iloc[:, [1]]
#titledf
length = df_new['Title'].str.len()
words = df_new['Title'].str.split().str.len()


# In[102]:


title_df['Title Length'] = length
title_df['Total Words'] = words


# In[103]:


title_df


# #### Character Length

# In[121]:


mean_length = title_df['Title Length'].mean()
mode_length = title_df['Title Length'].mode()
median_length = title_df['Title Length'].median()


# In[122]:


print('The mean of the titles length is ', mean_length)
print('The mode of the titles length is ', mode_length)
print('The media of the ttiles length is ', median_length)


# #### Title Word Length

# In[123]:


mean_word = title_df['Total Words'].mean()
mode_word = title_df['Total Words'].mode()
median_word = title_df['Total Words'].median()


# In[124]:


print('The mean of the titles word is ', mean_word)
print('The mode of the titles word is ', mode_word)
print('The media of the ttiles word is ', median_word)


# In[108]:


fig, ax = plt.subplots()
x = title_df['Title Length']
plt.title('Title Length Count Histogram', fontsize =20)

plt.hist(x, bins=20, color='g', edgecolor='k', alpha=0.65)

plt.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)


# In[105]:


fig = px.histogram(title_df, x="Title Length", nbins=20, title='Histogram showing the length of the title')
fig.show()


# In[106]:


fig = px.histogram(title_df, x="Total Words", nbins=20, title='Histogram showing the length of the title')
fig.show()


# #### Keywords

# In[2]:


import re
from collections import Counter


# In[ ]:





# In[ ]:





# In[ ]:




