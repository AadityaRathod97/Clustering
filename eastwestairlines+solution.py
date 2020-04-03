
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
air = pd.read_csv('EastWestAirlines.csv')


# In[2]:


air.head(10)


# In[3]:


air.describe()


# In[4]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[5]:


df_norm = norm_func(air.iloc[:,1:])
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


plt.figure(figsize=(30, 10))


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
h_complete	=	AgglomerativeClustering(n_clusters=7,	linkage='complete',affinity = "euclidean").fit(df_norm) 


# In[13]:


cluster_labels=pd.Series(h_complete.labels_)
air['clust']=cluster_labels # creating a  new column and assigning it to new column 
air = air.iloc[:,1:]
air.head()

