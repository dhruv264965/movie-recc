import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading ratings file
ratings = pd.read_csv('F:\Movie_Recc/csv_files/ratings.csv', 
                      sep='\t', 
                      encoding='latin-1', 
                      usecols=['user_id', 'movie_id', 'rating'])

# Reading users file
users = pd.read_csv('F:\Movie_Recc/csv_files/users.csv', 
                    sep='\t', encoding='latin-1', 
                    usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
movies = pd.read_csv('F:\Movie_Recc/csv_files/movies.csv', 
                     sep='\t', encoding='latin-1', 
                     usecols=['movie_id', 'title', 'genres'])

# file info
print(ratings.info())

print(users.info())

print(movies.info())

import wordcloud
from wordcloud import WordCloud, STOPWORDS

movies['title'] = movies['title'].fillna("").astype('str')
title_corpus = ' '.join(movies['title'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)

# Plot the wordcloud
plt.figure(figsize=(10,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()

title_corpus

# Get summary statistics of rating
ratings['rating'].describe()

ratings_median = ratings['rating'].median()

# Import seaborn library
import seaborn as sns
sns.set_style('whitegrid')
sns.set(font_scale=1.25)


# Display distribution of rating
sns.displot(ratings['rating'].fillna(ratings_median))

# merge all three datasets
merged_dataset = pd.merge(pd.merge(movies,ratings),users)
# 20 movies with highest ratings
merged_dataset[['title','genres','rating']].sort_values('rating',ascending = False).head(20)

# Make a census of the genre keywords
genre_labels = set()
for s in movies['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))
    
print("",genre_labels)

# function for counting the number of time genre keyword occurs
def genre_count(dataset, census, col):
    word_count = dict()
    for s in census:
        word_count[s] = 0
    for word in dataset[col].str.split('|'):
        if type(word) == float and pd.isnull(word): 
            continue
        for s in [s for s in word if s in census]: 
            if pd.notnull(s): 
                word_count[s] += 1 
    keyword_occurences = []
    for k,v in word_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, word_count

# Calling this function gives access to a list of genre keywords which are sorted by decreasing frequency
keyword_occurences, dum = genre_count(movies, genre_labels, 'genres')
keyword_occurences[:5]

# Define the dictionary used to produce the genre wordcloud
genres = dict()
trunc_occurences = keyword_occurences[0:18]
for s in trunc_occurences:
    genres[s[0]] = s[1]
    
# Create the wordcloud
genre_wc = WordCloud(width=1000,height=400, background_color='white')
genre_wc.generate_from_frequencies(genres)

# Plot the wordcloud
plt.figure(figsize=(10,8))
plt.imshow(genre_wc)
plt.axis('off')
plt.show()

# Break up the big genre string into a string array
movies['genres'] = movies['genres'].str.split('|')
# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')

movies['genres']

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
# Create the transform
tf = TfidfVectorizer(analyzer ='word', ngram_range = (1,2), min_df = 0, stop_words='english')
# tokenize ,build vocab and encode document
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape

print(len(tf.vocabulary_))

print(tf.idf_.shape)

print(tfidf_matrix.toarray())

# Cosine Similarity using linear kernel
# tf idf functionality produces normalized vectors, in which case cosine_similarity is equivalent to linear_kernel
from sklearn.metrics.pairwise import linear_kernel
cosine_sim_tfidf = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim_tfidf[:4, :4]

cosine_sim_tfidf.shape

# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def recommend_movies_tfidf(title):
  idx = indices[title]
  sorted_arr = np.argsort(cosine_sim_tfidf[idx])
  sorted_arr = sorted_arr[-21:-1]
  return titles.iloc[sorted_arr]
indices
# Recommendation using TF-IDF technique
recommend_movies_tfidf('Toy Story (1995)').head(20)

# Recommendation using TF-IDF technique
recommend_movies_tfidf('Good Will Hunting (1997)').head(20)

# Recommendation using TF-IDF technique
recommend_movies_tfidf('Saving Private Ryan (1998)').head(20)

# Count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Create a Vectorizer Object
vectorizer = CountVectorizer()
# Encode the Document  
cv_matrix = vectorizer.fit_transform(movies['genres'])
arr_CountVec = cv_matrix.toarray()

cv_matrix.shape

print("Vocabulary: ", len(vectorizer.vocabulary_))
print("Vocabulary: ", vectorizer.vocabulary_)
print("Length of Encoded Document is:", cv_matrix.shape)
print("Encoded Document is:", arr_CountVec)

# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim_cv = cosine_similarity(cv_matrix, cv_matrix)
cosine_sim_cv[:4, :4]

cosine_sim_cv.shape
# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

# function to recommend movies (CountVectorizer)
def recommend_movies_cv(title):
  idx = indices[title]
  sorted_arr = np.argsort(cosine_sim_cv[idx])
  sorted_arr = sorted_arr[-21:-1]
  return titles.iloc[sorted_arr]

# Recommendation using CountVectorizer technique
recommend_movies_cv('Toy Story (1995)')
print(set(recommend_movies_cv('Toy Story (1995)'))-set(recommend_movies_tfidf('Toy Story (1995)')))
print(set(recommend_movies_tfidf('Toy Story (1995)'))- set(recommend_movies_cv('Toy Story (1995)')))

# Fill Nan values in userid and movieid column with 0
ratings['movie_id'] = ratings['movie_id'].fillna(0)
ratings['user_id'] = ratings['user_id'].fillna(0)

# Randomly sample 2% of the ratings dataset
small_dataset = ratings.sample(frac=0.03)
# Check the sample info
print(small_dataset.info())

# split into training and testing 
from sklearn.model_selection import train_test_split as tts
train_data, test_data = tts(small_dataset, test_size = 0.2, shuffle = True)

# Creating two user matrix, one for testing and other for training
train_data_matrix = train_data[['user_id', 'movie_id', 'rating']].to_numpy()
test_data_matrix = test_data[['user_id', 'movie_id', 'rating']].to_numpy()

# Check their shape
print(train_data_matrix.shape)
print(test_data_matrix.shape)

from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation[:4, :4])

# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation[:4, :4])

# Function to predict ratings
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

from sklearn.metrics import mean_squared_error
from math import sqrt

# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))

# Predict ratings on the training data with both similarity score
user_prediction = predict(train_data_matrix, user_correlation, type='user')
item_prediction = predict(train_data_matrix, item_correlation, type='item')

# RMSE on the test data
print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

# RMSE on the train data
print('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))