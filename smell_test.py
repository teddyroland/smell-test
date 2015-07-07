ig_path =         # e.g. "/Users/[username]/Iterating Grace/Iterating Grace.txt"
suspect_path =    # e.g. "/Users/[username]/Iterating Grace/suspects/"

# The paths in this script are set up assuming that there is a folder containing
# .txt files for only the suspects' texts, and that there is a separate location
# for the unknown text. The names of suspect's .txt files will be used to
# identify them in the pandas dataframe and visualizations.


def smell_test(uk_path=ig_path, suspect_path=suspect_path, slices = "random", bag_size = None, num_samples = 100):
    # slices determines whether tokens are chosen from texts randomly ("random") or from
    # continuous passages ("passage"); bag_size determines how many tokens to retrieve
    # for each slice; by default, it will use the number of tokens in the unknown text;
    # trying to call more can enter an infinite while loop

    import string
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from random import randint
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import MDS
    import matplotlib.patches as mpatches
    from nltk.tokenize import word_tokenize


    # pronoun list to remove from text
    engl_pron = ['i', 'me','my', 'mine','myself','you', 'your','yours','yourself',\
                'she', 'her', 'hers', 'herself', 'he', 'him', 'his', 'himself',\
                'it', 'its', 'itself', 'we','us','our', 'ours','ourselves',\
                'yourselves', 'they','them','their', 'theirs', 'themselves',\
                'im', 'youre', 'youll','youd', 'shes', 'hes', 'hed', 'theyre', 'theyd']

    # takes list of tokens, returns string of randomly selected tokens separated by space
    # (sklearn's "Vectorizer" likes to take a string as its input)
    def random_bag(tokens, bag_size):
        auth_bag = []
        counter = []
        while len(counter)<bag_size:
            j = randint(0,len(tokens)-1)
            if j not in counter:
                auth_bag.append(tokens[j])
                counter.append(j)
        return " ".join(auth_bag)

    # same as random_bag, but returns continuous passages of text
    def passage_bag(tokens, bag_size):
        j = randint(0,len(tokens)-bag_size)
        auth_bag = tokens[j:j+bag_size]
        return " ".join(auth_bag)


    ## IMPORT TEXTS

    # text of "IG"
    with open(uk_path) as f:
        uk_string = f.read()

    # text of "IG" tokenized (quick and dirty)
    # Note: the first step is stripping non-ascii characters, bc python 2.X doesn't like unicode
    uk_tokens = [x for x in word_tokenize("".join([x for x in uk_string if ord(x)<128]))\
                 if x not in string.punctuation]

    uk_label = "Unknown"
    
    if bag_size == None:
        if len(uk_tokens) < 5000:
            bag_size = len(uk_tokens)
        else:
            bag_size = 5000

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

    # Slices texts into num_samples bags, produces feature arrays
    # takes a while to run on my computer

    bag_list = [] # each list item will be a string of randomly chosen words from suspect's writing
    labels = [] # keeps track of which string belongs to which author
    
    if slices == "random":
        for i in range(len(auth_frame['TOKENS'])):
            for j in range(num_samples):
                bag_list.append(random_bag(auth_frame['TOKENS'][i],bag_size=bag_size))
                labels.append(i)
        uk_bag = [random_bag(uk_tokens,bag_size=bag_size)]


    
    elif slices == "passage":
        for i in range(len(auth_frame['TOKENS'])):
            for j in range(num_samples):
                bag_list.append(passage_bag(auth_frame['TOKENS'][i],bag_size=bag_size))
                labels.append(i)
        uk_bag = [passage_bag(uk_tokens,bag_size=bag_size)]

        
    # Vectorizer will take the list of strings and return a sparse matrix with the feature set        

    # CountVectorizer is where account for Eder's and Hoover's observations
    # remove pronouns; rule of thumb # of fts 500; min_df suggested at 0.6-0.8 (for large # suspects)
    v = CountVectorizer(stop_words=engl_pron, max_features=500) #min_df=0.6
    bag_mtrx = v.fit_transform(bag_list)
    grc_mtrx = v.transform(uk_bag)
    
    # many functions prefer arrays to sparse matrices, and with # of suspects < 10 and n (slices) < 1000,
    # we don't have so much data that an array is unwieldy
    bag_array = bag_mtrx.toarray()
    grc_array = grc_mtrx.toarray()


    ## COSINE DISTANCES & VISUALIZATION

    mds = MDS(n_components=1)

    for i in range(len(auth_frame['AUTH_NAME'])):
        new_array = np.concatenate((bag_array[i*num_samples:(i+1)*num_samples],grc_array), axis=0)

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
        red_patch = mpatches.Patch(color='r', label=uk_label)
        h_list = [mpatches.Patch(color='b', label = auth_frame['AUTH_NAME'][i])]
        h_list.append(red_patch)

        #plt.legend(handles=h_list,loc=0)
        plt.legend(handles=h_list, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.show()
        print
