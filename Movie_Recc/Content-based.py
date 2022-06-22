# Import packages
import os
import pandas as pd

#file directories
user_file = 'F:\Movie_Recc/dat_files/users.dat'
rating_file = 'F:\Movie_Recc/dat_files/ratings.dat'
movie_file = 'F:\Movie_Recc/dat_files/movies.dat'

# Specify User's Age and Occupation Column
ages = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
occupations = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }

# Define csv files to be saved into
users_csv_file = 'F:\Movie_Recc/csv_files/users.csv'
movies_csv_file = 'F:\Movie_Recc/csv_files/movies.csv'
ratings_csv_file = 'F:\Movie_Recc/csv_files/ratings.csv'

# Read the Ratings File
ratings = pd.read_csv("F:\Movie_Recc/dat_files/ratings.dat", 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['user_id', 'movie_id', 'rating', 'timestamp'])                                                                                                                                                       

print("",ratings.head())
max_userid = ratings['user_id'].drop_duplicates().max()
max_movieid = ratings['movie_id'].drop_duplicates().max()

#prcessing for Keras deep learning model
# Add user_emb_id column whose values == user_id - 1
ratings['user_emb_id'] = ratings['user_id']-1
# Add movie_emb_id column whose values == movie_id - 1
ratings['movie_emb_id'] = ratings['movie_id'] - 1

print (len(ratings), 'ratings loaded')

# Save into ratings.csv
ratings.to_csv(ratings_csv_file, 
               sep='\t', 
               header=True, 
               encoding='latin-1', 
               columns=['user_id', 'movie_id', 'rating', 'timestamp', 'user_emb_id', 'movie_emb_id'])
print ('Saved to', ratings_csv_file)

#read users file
users = pd.read_csv("F:\Movie_Recc/dat_files/users.dat", 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
print("",users.head())

users['age_desc'] = users['age'].apply(lambda x: ages[x])
users['occ_desc'] = users['occupation'].apply(lambda x: occupations[x])
print (len(users), 'descriptions of', max_userid, 'users loaded.')

# Save into users.csv
users.to_csv(users_csv_file, 
             sep='\t', 
             header=True, 
             encoding='latin-1',
             columns=['user_id', 'gender', 'age', 'occupation', 'zipcode', 'age_desc', 'occ_desc'])
print ('Saved to', users_csv_file)

# Read the Movies File
movies = pd.read_csv(movie_file, 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['movie_id', 'title', 'genres'])
print (len(movies), 'descriptions of', max_movieid, 'movies loaded.')


movies.head()

# Save into movies.csv
movies.to_csv(movies_csv_file, 
              sep='\t', 
              header=True, 
              columns=['movie_id', 'title', 'genres'])
print ('Saved to', movies_csv_file)