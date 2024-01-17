#!/usr/bin/env python
# coding: utf-8

# # Activity: Build a K-means model 
# 
# ## **Introduction**
# 
# K-means clustering is very effective when segmenting data and attempting to find patterns. Because clustering is used in a broad array of industries, becoming proficient in this process will help you expand your skillset in a widely applicable way.   
# 
# In this activity, you are a consultant for a scientific organization that works to support and sustain penguin colonies. You are tasked with helping other staff members learn more about penguins in order to achieve this mission. 
# 
# The data for this activity is in a spreadsheet that includes datapoints across a sample size of 345 penguins, such as species, island, and sex. Your will use a K-means clustering model to group this data and identify patterns that provide important insights about penguins.
# 
# **Note:** Because this lab uses a real dataset, this notebook will first require basic EDA, data cleaning, and other manipulations to prepare the data for modeling. 

# ## **Step 1: Imports** 
# 

# Import statements including `K-means`, `silhouette_score`, and `StandardScaler`.

# In[1]:


# Import standard operational packages.
import pandas as pd
import numpy as np

# Important tools for modeling and evaluation.
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import visualization packages.
import seaborn as sns
import matplotlib.pyplot as plt


### YOUR CODE HERE ###


# `Pandas` is used to load the penguins dataset, which is built into the `seaborn` library. The resulting `pandas` DataFrame is saved in a variable named `penguins`. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA. 

# Save the `pandas` DataFrame in variable `penguins`. 

### YOUR CODE HERE ###

penguins = pd.read_csv("penguins.csv")


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `load_dataset` function. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The function is from seaborn (`sns`). It should be passed in the dataset name `'penguins'` as a string. 
# 
# </details>

# Now, review the first 10 rows of data.
# 

# In[3]:


# Review the first 10 rows.

### YOUR CODE HERE ###
penguins.head(10)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `head()` method.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# By default, the method only returns five rows. To change this, specify how many rows `(n = )` you want.
# 
# </details>

# ## **Step 2: Data exploration** 
# 
# After loading the dataset, the next step is to prepare the data to be suitable for clustering. This includes: 
# 
# *   Exploring data
# *   Checking for missing values
# *   Encoding data 
# *   Dropping a column
# *   Scaling the features using `StandardScaler`

# ### Explore data
# 
# To cluster penguins of multiple different species, determine how many different types of penguin species are in the dataset.

# In[4]:


# Find out how many penguin types there are.

### YOUR CODE HERE ###
penguins['species'].unique()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `unique()` method.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `unique()` method on the column `'species'`.
# 
# </details>

# In[5]:


# Find the count of each species type.

### YOUR CODE HERE ###
penguins['species'].value_counts()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `value_counts()` method.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `value_counts()` method on the column `'species'`.
# 
# </details>

# **Question:** How many types of species are present in the dataset?

# There are three types of penguin species in the data set.

# **Question:** Why is it helpful to determine the perfect number of clusters using K-means when you already know how many penguin species the dataset contains?

# It might help us to evaluate how well the model clusters/labels penguins based on their characteristics. Also while inertia and silhouette score might be better for the number of clusters different than 3 it may not be very meaningful.

# ### Check for missing values

# An assumption of K-means is that there are no missing values. Check for missing values in the rows of the data. 

# In[6]:


# Check for missing values.

### YOUR CODE HERE ###
penguins.isna().sum()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `isnull` and `sum` methods. 
# 
# </details>

# Now, drop the rows with missing values and save the resulting pandas DataFrame in a variable named `penguins_subset`.

# In[7]:


# Drop rows with missing values.
# Save DataFrame in variable `penguins_subset`.

### YOUR CODE HERE ###
penguins_subset = penguins.dropna(axis = 0)

penguins.info()
#penguins_subset.info()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `dropna`. Note that an axis parameter passed in to this function should be set to 0 if you want to drop rows containing missing values or 1 if you want to drop columns containing missing values. Optionally, `reset_index` may also be used to avoid a SettingWithCopy warning later in the notebook. 
# </details>

# Next, check to make sure that `penguins_subset` does not contain any missing values.

# In[8]:


# Check for missing values.

### YOUR CODE HERE ###
penguins_subset.isna().sum()
#penguins_subset.info()


# Now, review the first 10 rows of the subset.

# In[9]:


# View first 10 rows.

### YOUR CODE HERE ###
penguins_subset.head(10)


# ### Encode data
# 
# Some versions of the penguins dataset have values encoded in the sex column as 'Male' and 'Female' instead of 'MALE' and 'FEMALE'. The code below will make sure all values are ALL CAPS. 
# 

# In[10]:


penguins_subset['sex'] = penguins_subset['sex'].str.upper()


# K-means needs numeric columns for clustering. Convert the categorical column `'sex'` into numeric. There is no need to convert the `'species'` column because it isn't being used as a feature in the clustering algorithm. 

# In[11]:


# Convert `sex` column from categorical to numeric.

### YOUR CODE HERE ###
penguins_subset['sex'] = pd.get_dummies(penguins_subset['sex'], drop_first = True)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `get_dummies` function. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `drop_first` parameter should be set to `True`. This removes redundant data. The `columns` parameter can **optionally** be set to `['sex']` to specify that only the `'sex'` column gets this operation performed on it. 
# 
# </details>

# ### Drop a column
# 
# Drop the categorical column `island` from the dataset. While it has value, this notebook is trying to confirm if penguins of the same species exhibit different physical characteristics based on sex. This doesn't include location.
# 
# Note that the `'species'` column is not numeric. Don't drop the `'species'` column for now. It could potentially be used to help understand the clusters later. 

# In[12]:


# Drop the island column.

### YOUR CODE HERE ###
penguins_subset.drop(columns = 'island', axis = 1, inplace = True)
penguins_subset.head()


# ### Scale the features
# 
# Because K-means uses distance between observations as its measure of similarity, it's important to scale the data before modeling. Use a third-party tool, such as scikit-learn's `StandardScaler` function. `StandardScaler` scales each point xᵢ by subtracting the mean observed value for that feature and dividing by the standard deviation:
# 
# x-scaled = (xᵢ – mean(X)) / σ
# 
# This ensures that all variables have a mean of 0 and variance/standard deviation of 1. 
# 
# **Note:** Because the species column isn't a feature, it doesn't need to be scaled. 
# 
# First, copy all the features except the `'species'` column to a DataFrame `X`. 

# In[13]:


# Exclude `species` variable from X

### YOUR CODE HERE ###
X = penguins_subset.copy()
X.drop(columns = 'species', axis = 1, inplace = True)
X.head()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use`drop()`.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Select all columns except `'species'.`The `axis` parameter passed in to this method should be set to `1` if you want to drop columns.
# </details>

# Scale the features in `X` using `StandardScaler`, and assign the scaled data to a new variable `X_scaled`. 

# In[14]:


#Scale the features.
#Assign the scaled data to variable `X_scaled`.

### YOUR CODE HERE ###
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_scaled


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Instantiate StandardScaler to transform the data in a single step.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `.fit_transform()` method and pass in the data as an argument.
# </details>

# ## **Step 3: Data modeling** 

# Now, fit K-means and evaluate inertia for different values of k. Because you may not know how many clusters exist in the data, start by fitting K-means and examining the inertia values for different values of k. To do this, write a function called `kmeans_inertia` that takes in `num_clusters` and `x_vals` (`X_scaled`) and returns a list of each k-value's inertia.
# 
# When using K-means inside the function, set the `random_state` to `42`. This way, others can reproduce your results.

# In[30]:


# Fit K-means and evaluate inertia for different values of k.

### YOUR CODE HERE ###
def kmeans_inertia(num_clusters, x_vals):
        
    inertia = []
    
    for i in num_clusters:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(x_vals)
        inertia.append(kmeans.inertia_)
    return inertia


# Use the `kmeans_inertia` function to return a list of inertia for k=2 to 10.

# In[31]:


# Return a list of inertia for k=2 to 10.

### YOUR CODE HERE ###
num_clusters = list(range(2,11))

inertia = kmeans_inertia(num_clusters, X_scaled)
inertia


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review the material about the `kmeans_inertia` function. 
# </details>

# Next, create a line plot that shows the relationship between `num_clusters` and `inertia`.
# Use either seaborn or matplotlib to visualize this relationship. 

# In[66]:


# Create a line plot.

### YOUR CODE HERE ###
sns.lineplot(x = num_clusters, y = inertia);
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Chart for the K-Means Model');


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `sns.lineplot`.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Include `x=num_clusters` and `y=inertia`.
# </details>

# **Question:** Where is the elbow in the plot?

# The elbow is when the cluster number is 5 or even 6.

# ## **Step 4: Results and evaluation** 

# Now, evaluate the silhouette score using the `silhouette_score()` function. Silhouette scores are used to study the distance between clusters. 

# Then, compare the silhouette score of each value of k, from 2 through 10. To do this, write a function called `kmeans_sil` that takes in `num_clusters` and `x_vals` (`X_scaled`) and returns a list of each k-value's silhouette score.

# In[67]:


# Evaluate silhouette score.
# Write a function to return a list of each k-value's score.

### YOUR CODE HERE ###
def kmeans_sil(num_clusters,x_vals):
    sil_score = []
    for i in num_clusters:
        kmeans = KMeans(n_clusters = i, random_state = 42)
        kmeans.fit(x_vals)
        sil_score.append(silhouette_score(x_vals, kmeans.labels_))
    return sil_score

silhouette = kmeans_sil(num_clusters,X_scaled)
silhouette


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review the `kmeans_sil` function video.
# </details>

# Next, create a line plot that shows the relationship between `num_clusters` and `sil_score`.
# Use either seaborn or matplotlib to visualize this relationship. 

# In[69]:


# Create a line plot.

### YOUR CODE HERE ###
plot = sns.lineplot(x = num_clusters, y = silhouette, marker = 'o')
plot.set_title('Silhouette Score by Number of Clusters')
plot.set_xlabel('Number of Clusters')
plot.set_ylabel('Silhouette Score');


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `sns.lineplot`.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Include `x=num_clusters` and `y=sil_score`.
# </details>

# **Question:** What does the graph show?

# The silhouette score is maximized when the number of clusters is set to 6.

# ### Optimal k-value

# To decide on an optimal k-value, fit a six-cluster model to the dataset. 

# In[40]:


# Fit a 6-cluster model.

### YOUR CODE HERE ###
kmeans6 = KMeans(n_clusters = 6, random_state = 42)
kmeans6.fit(X_scaled)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Make an instance of the model with `num_clusters = 6` and use the `fit` function on `X_scaled`. 
# </details>
# 
# 
# 

# Print out the unique labels of the fit model.

# In[45]:


# Print unique labels.

### YOUR CODE HERE ###
np.unique(kmeans6.labels_)


# Now, create a new column `cluster` that indicates cluster assignment in the DataFrame `penguins_subset`. It's important to understand the meaning of each cluster's labels, then decide whether the clustering makes sense. 
# 
# **Note:** This task is done using `penguins_subset` because it is often easier to interpret unscaled data.

# In[47]:


# Create a new column `cluster`.

### YOUR CODE HERE ###
penguins_subset['cluster'] = kmeans6.labels_
penguins_subset.head()


# Use `groupby` to verify if any `'cluster'` can be differentiated by `'species'`.

# In[59]:


# Verify if any `cluster` can be differentiated by `species`.

### YOUR CODE HERE ###
penguins_subset.groupby(by = ['cluster','species']).size()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `groupby(by=['cluster', 'species'])`. 
# 
# </details>
# 

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# 
# Use an aggregation function such as `size`.
# 
# </details>

# Next, interpret the groupby outputs. Although the results of the groupby show that each `'cluster'` can be differentiated by `'species'`, it is useful to visualize these results. The graph shows that each `'cluster'` can be differentiated by `'species'`. 
# 
# **Note:** The code for the graph below is outside the scope of this lab. 

# In[58]:


penguins_subset.groupby(by=['cluster', 'species']).size().plot.bar(title='Clusters differentiated by species',
                                                                   figsize=(6, 5),
                                                                   ylabel='Size',
                                                                   xlabel='(Cluster, Species)');


# Use `groupby` to verify if each `'cluster'` can be differentiated by `'species'` AND `'sex_MALE'`.

# In[64]:


# Verify if each `cluster` can be differentiated by `species' AND `sex_MALE`.

### YOUR CODE HERE ###
penguins_subset.groupby(by = ['cluster','species', 'sex']).size()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `groupby(by=['cluster','species', 'sex_MALE'])`. 
# </details>
# 

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use an aggregation function such as `size`.
# </details>

# **Question:** Are the clusters differentiated by `'species'` and `'sex_MALE'`?

# Yes. The male and female adelie penguins are mostly assigned to cluster 0 and cluster 2, respectively. The exceptions were two adelie males that were put into a different cluster, cluster 4. Cluster 4 was mostly male chinstrap penguins. The female chinstrap penguins were mostly assigned to cluster 5. However, 5 of them were erroneously assigned to the adelie female pengiuns cluster, cluster 2. This suggests that adelie and chinstrap penguins might be similar in certain characteristics.
# 
# Gentoo penguins had their own exclusive clusters. The females were assigned to cluster 1 and the males to cluster 3.

# Finally, interpret the groupby outputs and visualize these results. The graph shows that each `'cluster'` can be differentiated by `'species'` and `'sex_MALE'`. Furthermore, each cluster is mostly comprised of one sex and one species. 
# 
# **Note:** The code for the graph below is outside the scope of this lab. 

# In[65]:


penguins_subset.groupby(by=['cluster','species','sex']).size().unstack(level = 'species', fill_value=0).plot.bar(title='Clusters differentiated by species and sex',
                                                                                                                      figsize=(6, 5),
                                                                                                                      ylabel='Size',
                                                                                                                      xlabel='(Cluster, Sex)')
plt.legend(bbox_to_anchor=(1.3, 1.0))


# ## **Considerations**
# 
# 
# **What are some key takeaways that you learned during this lab? Consider the process you used, key tools, and the results of your investigation.**
# 
# - Kmeans is a useful tool to separate unlabeled data into meaningful clusters.
# - Inertia and silhouette scores are useful metrics to identify the number of clusters.
# - Even if we know the number of types/categories/clusters apriori, using the elbow method and silhouette scores can help us to discover more useful categorization.
# 
# **What summary would you provide to stakeholders?**
# 
# Penguins can be meaningfully clustered by species and sex. So for the current dataset, it is useful to have 6 clusters (3 species times 2 sex categfories). When the cluster number is set to 6, the kmeans model seems to perform the best. This is confirmed by elbow method. Moreover, the silhouette score is maximized at cluster number being 6. 
# 
# 
# 

# ### References
# 
# [Gorman, Kristen B., et al. “Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis).” PLOS ONE, vol. 9, no. 3, Mar. 2014, p. e90081. PLoS Journals](https://doi.org/10.1371/journal.pone.0090081)
# 
# [Sklearn Preprocessing StandardScaler scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged
