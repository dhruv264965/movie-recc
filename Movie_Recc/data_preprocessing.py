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