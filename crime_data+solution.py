
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
crime = pd.read_csv('crime_data.csv')


# In[2]:


crime


# In[3]:


crime.describe()


# In[4]:


def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)


# In[5]:


df_norm = norm_func(crime.iloc[:,1:])
df_norm


# In[6]:


type(df_norm)


# In[7]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 


# In[8]:


z = linkage(df_norm, method="complete",metric="euclidean")


# In[9]:


z


# In[10]:


plt.figure(figsize=(15, 5))


# In[11]:


plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[12]:


# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 


# In[14]:


cluster_labels=pd.Series(h_complete.labels_)
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,1:]
crime.head(100)

