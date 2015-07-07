# full code supporting this blog post:
# http://teddyroland.com/2015/07/02/attributing-authorship-to-iterating-grace-or-the-smell-test-of-style/


from nltk.tokenize import word_tokenize
import string
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from random import randint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from matplotlib.colors import ColorConverter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.patches as mpatches

ig_path = "/Users/edwinroland/Desktop/Iterating Grace/Iterating Grace.txt"
suspect_path = "/Users/edwinroland/Desktop/Iterating Grace/suspects/"

# The paths in this script are set up assuming that there is a folder containing
# .txt files for only the suspects' texts, and that there is a separate location
# for "Iterating Grace.txt". The names of suspect's .txt files will be used to
# identify them in the pandas dataframe and visualizations.


# pronoun list to remove from text
engl_pron = ['i', 'me','my', 'mine','myself','you', 'your','yours','yourself',\
            'she', 'her', 'hers', 'herself', 'he', 'him', 'his', 'himself',\
            'it', 'its', 'itself', 'we','us','our', 'ours','ourselves',\
            'yourselves', 'they','them','their', 'theirs', 'themselves',\
            'im', 'youre', 'youll','youd', 'shes', 'hes', 'hed', 'theyre', 'theyd']

# takes list of tokens, returns string of randomly selected tokens separated by space
# (sklearn's "Vectorizer" likes to take a string as its input)
def random_bag(tokens, bag_size = 1971): # under this method, there are 1971 tokens in IG
    auth_bag = []
    counter = []
    while len(counter)<bag_size:
        j = randint(0,len(tokens)-1)
        if j not in counter:
            auth_bag.append(tokens[j])
            counter.append(j)
    return " ".join(auth_bag)

# same as random_bag, but returns continuous passages of text
def passage_bag(tokens, bag_size = 1971): # under this method, there are 1971 tokens in IG
    j = randint(0,len(tokens)-bag_size)
    auth_bag = tokens[j:j+bag_size]
    return " ".join(auth_bag)


# finds interquartile range of a list of numbers; handy for stats
def iqr(dist):
    q75, q25 = np.percentile(dist, [75 ,25])
    return q75 - q25

## IMPORT TEXTS

# text of "IG"
with open(ig_path) as f:
    iterating_string = f.read()

# text of "IG" tokenized (quick and dirty)
# Note: the first step is stripping non-ascii characters, bc python 2.X doesn't like unicode
ig_tokens = [x for x in word_tokenize("".join([x for x in iterating_string if ord(x)<128]))\
             if x not in string.punctuation]

# import each suspect's text into a dataframe, tokenizes text as well

auth_frame = pd.DataFrame()
auth_name = []
auth_text = []
for x in [x for x in os.listdir(suspect_path) if x[-4:]=='.txt']:
    auth_name.append(x[:-4])
    with open(suspect_path+x) as f:
        auth_text.append(f.read())
auth_frame['AUTH_NAME'] = auth_name # This column is really just the filenames
auth_frame['TEXT'] = auth_text
auth_frame['TOKENS'] = [[x for x in word_tokenize("".join([x for x in auth_text[i] if ord(x)<128]))\
                         if x not in string.punctuation] for i in range(len(auth_text))]

## DATA SAMPLING

# Slices texts into n random bags, returns feature arrays
# takes a while to run on my computer

n = 100 # number of slices to take from each text
bag_list = [] # each list item will be a string of randomly chosen words from suspect's writing
labels = [] # keeps track of which string belongs to which author
for i in range(len(auth_frame['TOKENS'])):
    for j in range(n):
        bag_list.append(random_bag(auth_frame['TOKENS'][i]))
        #bag_list.append(passage_bag(auth_frame['TOKENS'][i])) # alternate function calls up passages
        labels.append(i)



## CLASSIFIER PIPELINE
        
# Vectorizer will take the list of strings and return a sparse matrix with the feature set        

# CountVectorizer is where account for Eder's and Hoover's observations
# remove pronouns; rule of thumb # of fts 500; min_df suggested at 0.6-0.8 (for large # suspects)
v = CountVectorizer(stop_words=engl_pron, max_features=500) #min_df=0.6
bag_mtrx = v.fit_transform(bag_list)
grc_mtrx = v.transform([random_bag(ig_tokens)])
# grc_mtrx = v.transform([random_bag(ig_tokens)]) # alternate function calls up passages

# many functions prefer arrays to sparse matrices, and with # of suspects < 10 and n (slices) < 1000,
# we don't have so much data that an array is unwieldy
bag_array = bag_mtrx.toarray()
grc_array = grc_mtrx.toarray()

## AUTHOR PREDICTION

# SVM classifier (per Eder's empirical findings, convenience in sklearn)
# reports overall accuracy on cross-validation, predicted author(s) of IG

clf = LinearSVC()

# I personally like to run a classifier multiple times and check the median score (and IQR),
# to get a sense of its robustness, so I have lists that collect each iteration's output

accr = [] # overall accuracy of classification
f1s = [] # F1 scores of classification
auth_preds = [] # each iteration's prediction for IG's authorship

for i in range(100):
    train_features, test_features, train_labels, test_labels = train_test_split(bag_array,
                                                                                labels, test_size=0.25)
    clf.fit(train_features, train_labels)
    predicted = clf.predict(test_features)
    
    accr.append(accuracy_score(test_labels, predicted))
    f1s.append(f1_score(test_labels, predicted))
    auth_preds.append(clf.predict(grc_array)[0])

print "Median(IQR)   Accuracy:",np.median(accr),"(", iqr(accr), ")", "  F1:", np.median(f1s), '(', iqr(f1s), ')'
print
print "Suspect List"
c = Counter(auth_preds)
for k in c.keys():
    # author name and its share of all predictions
    print auth_frame["AUTH_NAME"][k], int(100*float(c[k])/sum(c.values())), "%"

## PCA GRAPH
# all of this code replicates biplot() in R

# list of colors in MPL
colors = ColorConverter.cache.values()

# PCA, naturally
pca = PCA(n_components=4)

## project feature arrays into PC space
bag_pca = pca.fit_transform(bag_array)
grc_pca = pca.transform(grc_array)

# text information
# 0,1 denote PC1 and PC2; change values for other PCs
xs = bag_pca[:,0] # see 'prcomp(my_data)$x' in R
ys = bag_pca[:,1]

# loading information
xvector = pca.components_[0] # see 'prcomp(my_data)$rotation' in R
yvector = pca.components_[1]

dvec = np.array([np.sqrt(xvector[i]**2+yvector[i]**2) for i in range(len(xvector))])
load_ind = dvec.argsort()[-10:][::-1]

plt.figure(1, figsize=(10, 10), dpi=200)    

# plots suspects' texts
for i in range(len(xs)):
    plt.plot(xs[i], ys[i], 'x', color = colors[labels[i]+12])
    
# plots loadings
for i in range(len(load_ind)):
    plt.arrow(0, 0, xvector[load_ind[i]]*max(xs), yvector[load_ind[i]]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvector[load_ind[i]]*max(xs)*1.2, yvector[load_ind[i]]*max(ys)*1.2,
             list(v.get_feature_names())[load_ind[i]], color='r')

# This is inelegant, but I was having trouble making a pretty legend
import matplotlib.patches as mpatches

#grn_patch = mpatches.Patch(color='r', label='"Grace"')
grn_patch = mpatches.Patch(color=colors[len(auth_name)+12], label='"Grace"')
h_list = [mpatches.Patch(color=colors[i+12], label = auth_frame['AUTH_NAME'][i]) for i in range(len(auth_name))]
h_list.append(grn_patch)
plt.legend(handles=h_list,loc=0)

# plots IG
plt.plot(grc_pca[0][0], grc_pca[0][1],  'o', color=colors[len(auth_name)+12])

print 'Explained Variance', pca.explained_variance_ratio_[:2]
# Haven't yet put this info on the axes like biplot() does!

plt.show()

## COSINE SIMILARITY/MDS GRAPH
# biplot modded for MDS with Cosine Similarity

# MDS, naturally
mds = MDS(n_components=2)
new_array = np.concatenate((bag_array,grc_array), axis=0) # incl. IG vector with suspects'
distances = 1 - cosine_similarity(new_array)

# Scales Cosine Distances into 2-D space
new_mds = mds.fit_transform(new_array)

# how we'll distinguish IG from the suspects
mds_markers = ['x']*len(labels)+['o']
mds_labels = labels[:]
mds_labels.append(labels[-1]+1)

# coordinates of 2-D scaled slice vectors
xs = new_mds[:,0]
ys = new_mds[:,1]

# plotting points
plt.figure(1, figsize=(10, 10), dpi=200)    
for i in range(len(xs)):
    plt.plot(xs[i], ys[i], marker = mds_markers[i], color = colors[mds_labels[i]+12])

# legend
grn_patch = mpatches.Patch(color=colors[len(auth_name)+12], label='"Grace"')
h_list = [mpatches.Patch(color=colors[i+12], label = auth_frame['AUTH_NAME'][i]) for i in range(len(auth_name))]
h_list.append(grn_patch)
plt.legend(handles=h_list,loc=0)

plt.show()

## SMELL TEST

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.patches as mpatches

# MDS, naturally
mds = MDS(n_components=1)

for i in range(len(auth_frame['AUTH_NAME'])):
    new_array = np.concatenate((bag_array[i*n:(i+1)*n],grc_array), axis=0)
    # n here is the number of slices we decided to take earlier
    
    distances = 1 - cosine_similarity(new_array)

    # Scales Cosine Distances into 2-D space
    new_mds = mds.fit_transform(new_array)

    # how we'll distinguish IG from the suspects
    mds_markers = ['x']*(len(new_array)-1)+['o']
    colors = ['b']*(len(new_array)-1)+['r']

    # coordinates of 2-D scaled slice vectors
    xs = new_mds[:,0] # see 'prcomp(my_data)$x' in R
    ys = [0]*len(xs)

    # plotting points
    plt.figure(1, figsize=(10, 1))    
    plt.xlim(-75, 75) 
    for j in range(len(xs)):
        # plots suspects' texts
        plt.plot(xs[j], ys[j], marker = mds_markers[j], color = colors[j])

    # This is inelegant, but I was having trouble making a pretty legend
    red_patch = mpatches.Patch(color='r', label='"Grace"')
    h_list = [mpatches.Patch(color='b', label = auth_frame['AUTH_NAME'][i])]
    h_list.append(red_patch)

    #plt.legend(handles=h_list,loc=0)
    plt.legend(handles=h_list, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.show()
    print
