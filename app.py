import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import helper_functions as hf


train_data = pd.read_csv('data.csv')

train_data = hf.cleaning_data(train_data)

features = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo']
scaler = StandardScaler()

def recommend_songs(name, train_data):
    if name not in train_data['name'].values:
        names = hf.find_similar_tracks(name, train_data)
        for i, n in enumerate(names):
            response = input(f"Did you mean: {n}? (y/n) ").strip().lower()
            if response == 'y':
                name = n
                break
            elif response == 'n':
                if i == len(names) - 1:
                    print("No similar tracks found. Please try another name.")
                    return None
                continue
            else:
                print("Invalid response.")
                return None
    
    artist = hf.get_artists(name, train_data)
    if not artist:
        print("Artist selection failed. Please try again.")
        return None
    
    song_index = train_data[(train_data['name'] == name) & (train_data['artists'] == artist)].index[0]
    # Check if user wants to filter tracks
    response = input(f"Do you want to filter tracks based on explicit content, year, and popularity? (y/n) ").strip().lower()
    if response == 'y':
        filters = {
            'explicit': True,
            'year_range': 0,
            'popularity_range': 0
        }
        explicit_response = input("Do you want to exclude explicit content? (y/n) ").strip().lower()
        filters['explicit'] = explicit_response == 'n'
        
        year_response = input("Do you want to filter by year range? (Enter a number or 0 for no filtering): ")
        filters['year_range'] = int(year_response) if year_response.isdigit() else 0
        
        popularity_response = input("Do you want to filter by popularity range? (Enter a number or 0 for no filtering): ")
        filters['popularity_range'] = int(popularity_response) if popularity_response.isdigit() else 0
        
        X_filtered = hf.filter_tracks(train_data, filters, song_index)

        if X_filtered.empty:
            print("No tracks found after applying filters. Please try again with different filters.")
            return None
        
        # Check if the song exists in the filtered data
        if not ((X_filtered['name'] == name) & (X_filtered['artists'] == artist)).any():
            # Add the song to the filtered data
            song_data = train_data[(train_data['name'] == name) & (train_data['artists'] == artist)]
            X_filtered = pd.concat([X_filtered, song_data], ignore_index=True)

        model = NearestNeighbors(n_neighbors=6, metric='cosine')
        model.fit(scaler.fit_transform(X_filtered[features]))

        
        song_index = X_filtered[(X_filtered['name'] == name) & (X_filtered['artists'] == artist)].index[0]
        distances, indices = model.kneighbors(scaler.transform(X_filtered[features])[song_index].reshape(1, -1))
            
        print("------------------------------------------------")
        print(f"\nRecommendations for '{name}':")
        for i in range(1, len(distances[0])):
            recommended_song_index = indices[0][i]
            recommended_song = X_filtered.iloc[recommended_song_index]
            print(f"{i}. {recommended_song['name']} by {recommended_song['artists']} (Popularity: {recommended_song['popularity']}) Year: {recommended_song['year']} Explicit: {'Yes' if recommended_song['explicit'] else 'No'}")
        return song_index, indices[0][1:]

    else:
        model = NearestNeighbors(n_neighbors=6, metric='cosine')
        model.fit(scaler.fit_transform(train_data[features]))

    
        song_index = train_data[(train_data['name'] == name) & (train_data['artists'] == artist)].index[0]
        distances, indices = model.kneighbors(scaler.transform(train_data[features])[song_index].reshape(1, -1))

        print("------------------------------------------------")
        print(f"\nRecommendations for '{name}':")
        for i in range(1, len(distances[0])):
            recommended_song_index = indices[0][i]
            recommended_song = train_data.iloc[recommended_song_index]
            print(f"{i}. {recommended_song['name']} by {recommended_song['artists']} (Popularity: {recommended_song['popularity']}) Year: {recommended_song['year']} Explicit: {'Yes' if recommended_song['explicit'] else 'No'}")
        return song_index, indices[0][1:]
    
recommend_songs('hurt', train_data)