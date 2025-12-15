import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import helper_functions as hf
from sklearn.decomposition import PCA
import streamlit as st


train_data = pd.read_csv('data.csv')

train_data = hf.cleaning_data(train_data)

features = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo']
scaler = StandardScaler()

def recommend_songs(name, train_data):
    name = name.strip()
    if name not in train_data['name'].values:
        names = hf.find_similar_tracks(name, train_data) or []
        names = sorted(set(names))  # Remove duplicates + stable ordering
        if names:
            selected_name = st.selectbox("Select a similar track:", names)
            name = selected_name
        else:
            st.error("No similar tracks found. Please try another name.")
            return None
        # for i, n in enumerate(names):
        #     if st.radio(f"Did you mean: {n}?", options=['No', 'Yes'], key=f"radio_{i}") == 'Yes':
        #         name = n
        #         break
            # response = st.text_input(f"Did you mean: {n}? (y/n) ").strip().lower()
            # if response == 'y':
            #     name = n
            #     break
            # elif response == 'n':
            #     if i == len(names) - 1:
            #         print("No similar tracks found. Please try another name.")
            #         return None
            #     continue
            # else:
            #     print("Invalid response.")
            #     return None
        # else:
        #     st.warning("No similar tracks selected. Please try another name.")
        #     return None
        
    artists = hf.get_artists(name, train_data)
    artists = list(artists) if artists is not None else []
    if len(artists) == 0:
        st.error("Artist selection failed. Please try again.")
        return None
    artist = st.selectbox("Select the artist:", artists)
    
    song_index = train_data[(train_data['name'] == name) & (train_data['artists'] == artist)].index[0]
    # Check if user wants to filter tracks
    apply_filters = st.checkbox("Apply filters (explicit, year, popularity)", value=True)
    if apply_filters:
        filters = {
            'explicit': st.checkbox("Exclude explicit content", value=True),
            'year_range': st.number_input("Year range (0 for no filtering)", min_value=0, value=0),
            'popularity_range': st.number_input("Popularity range (0 for no filtering)", min_value=0, value=0)
        }
        
        X_filtered = hf.filter_tracks(train_data, filters, song_index)

        if X_filtered.empty:
            st.warning("No tracks found after applying filters. Please try again with different filters.")
            return None
        
        # Check if the song exists in the filtered data
        if not ((X_filtered['name'] == name) & (X_filtered['artists'] == artist)).any():
            # Add the song to the filtered data
            song_data = train_data[(train_data['name'] == name) & (train_data['artists'] == artist)]
            if song_data.empty:
                st.error("The selected song is not found in the dataset.")
                return None
            X_filtered = pd.concat([X_filtered, song_data], ignore_index=True)

        model = NearestNeighbors(n_neighbors=6, metric='cosine')
        X_scaled = scaler.fit_transform(X_filtered[features])
        model.fit(X_scaled)

        
        song_index = X_filtered[(X_filtered['name'] == name) & (X_filtered['artists'] == artist)].index[0]
        distances, indices = model.kneighbors(X_scaled[song_index].reshape(1, -1))
            
        st.write("------------------------------------------------")
        st.write(f"\nRecommendations for '{name}':")
        for i in range(1, len(distances[0])):
            recommended_song_index = indices[0][i]
            recommended_song = X_filtered.iloc[recommended_song_index]
            st.write(f"{i}. {recommended_song['name']} by {recommended_song['artists']} (Popularity: {recommended_song['popularity']}) Year: {recommended_song['year']} Explicit: {'Yes' if recommended_song['explicit'] else 'No'}")
        return song_index, indices[0][1:]
    
    else:
        model = NearestNeighbors(n_neighbors=6, metric='cosine')
        X_scaled = scaler.fit_transform(train_data[features])
        model.fit(X_scaled)

        distances, indices = model.kneighbors(X_scaled[song_index].reshape(1, -1))

        st.write("------------------------------------------------")
        st.write(f"\nRecommendations for '{name}':")
        for i in range(1, len(distances[0])):
            recommended_song_index = indices[0][i]
            recommended_song = train_data.iloc[recommended_song_index]
            st.write(f"{i}. {recommended_song['name']} by {recommended_song['artists']} (Popularity: {recommended_song['popularity']}) Year: {recommended_song['year']} Explicit: {'Yes' if recommended_song['explicit'] else 'No'}")
        return song_index, indices[0][1:]

    # response = input(f"Do you want to filter tracks based on explicit content, year, and popularity? (y/n) ").strip().lower()
    # if response == 'y':
    #     filters = {
    #         'explicit': True,
    #         'year_range': 0,
    #         'popularity_range': 0
    #     }
    #     explicit_response = input("Do you want to exclude explicit content? (y/n) ").strip().lower()
    #     filters['explicit'] = explicit_response == 'n'
        
    #     year_response = input("Do you want to filter by year range? (Enter a number or 0 for no filtering): ")
    #     filters['year_range'] = int(year_response) if year_response.isdigit() else 0
        
    #     popularity_response = input("Do you want to filter by popularity range? (Enter a number or 0 for no filtering): ")
    #     filters['popularity_range'] = int(popularity_response) if popularity_response.isdigit() else 0
        
    #     X_filtered = hf.filter_tracks(train_data, filters, song_index)

    #     if X_filtered.empty:
    #         print("No tracks found after applying filters. Please try again with different filters.")
    #         return None
        
    #     # Check if the song exists in the filtered data
    #     if not ((X_filtered['name'] == name) & (X_filtered['artists'] == artist)).any():
    #         # Add the song to the filtered data
    #         song_data = train_data[(train_data['name'] == name) & (train_data['artists'] == artist)]
    #         X_filtered = pd.concat([X_filtered, song_data], ignore_index=True)

    #     model = NearestNeighbors(n_neighbors=6, metric='cosine')
    #     model.fit(scaler.fit_transform(X_filtered[features]))

        
    #     song_index = X_filtered[(X_filtered['name'] == name) & (X_filtered['artists'] == artist)].index[0]
    #     distances, indices = model.kneighbors(scaler.transform(X_filtered[features])[song_index].reshape(1, -1))
            
    #     print("------------------------------------------------")
    #     print(f"\nRecommendations for '{name}':")
    #     for i in range(1, len(distances[0])):
    #         recommended_song_index = indices[0][i]
    #         recommended_song = X_filtered.iloc[recommended_song_index]
    #         print(f"{i}. {recommended_song['name']} by {recommended_song['artists']} (Popularity: {recommended_song['popularity']}) Year: {recommended_song['year']} Explicit: {'Yes' if recommended_song['explicit'] else 'No'}")
    #     return song_index, indices[0][1:]

    # else:
    #     model = NearestNeighbors(n_neighbors=6, metric='cosine')
    #     model.fit(scaler.fit_transform(train_data[features]))

    
    #     song_index = train_data[(train_data['name'] == name) & (train_data['artists'] == artist)].index[0]
    #     distances, indices = model.kneighbors(scaler.transform(train_data[features])[song_index].reshape(1, -1))

    #     print("------------------------------------------------")
    #     print(f"\nRecommendations for '{name}':")
    #     for i in range(1, len(distances[0])):
    #         recommended_song_index = indices[0][i]
    #         recommended_song = train_data.iloc[recommended_song_index]
    #         print(f"{i}. {recommended_song['name']} by {recommended_song['artists']} (Popularity: {recommended_song['popularity']}) Year: {recommended_song['year']} Explicit: {'Yes' if recommended_song['explicit'] else 'No'}")
    #     return song_index, indices[0][1:]

# Plot recommended songs in 2D space
def plot_recommendations(song_index, train_data, recommended_indices):
    pca = PCA(n_components=2)
    X_scaled = scaler.fit_transform(train_data[features])
    X_pca = pca.fit_transform(X_scaled)
    train_data['pca_x'] = X_pca[:, 0]
    train_data['pca_y'] = X_pca[:, 1]

    # plt.figure(figsize=(12, 8))
    # plt.scatter(train_data['pca_x'], train_data['pca_y'], alpha=0.2, color='gray', label='All Songs')

    # plt.scatter(train_data.iloc[song_index]['pca_x'], train_data.iloc[song_index]['pca_y'], color='blue', label='Selected Song', s=100)
    # if recommended_indices is not None:
    #     recommended_songs = train_data.iloc[recommended_indices]
    #     plt.scatter(recommended_songs['pca_x'], recommended_songs['pca_y'], color='red', label='Recommended Songs')

    # plt.title('Recommended Songs in 2D PCA Space')
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.legend()
    # plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(train_data['pca_x'], train_data['pca_y'], alpha=0.2, color='gray', label='All Songs')

    ax.scatter(train_data.iloc[song_index]['pca_x'], train_data.iloc[song_index]['pca_y'], color='blue', label='Selected Song', s=100)
    if recommended_indices is not None:
        recommended_songs = train_data.iloc[recommended_indices]
        ax.scatter(recommended_songs['pca_x'], recommended_songs['pca_y'], color='red', label='Recommended Songs')
    
    ax.set_title('Recommended Songs in 2D PCA Space')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# Streamlit app setup
st.set_page_config(page_title="Music Recommendation System", layout="wide")

st.title("Music Recommendation System")

# User input for song name
song_name = st.text_input("Enter a song name: ")
if song_name:
    st.write(f"Searching for recommendations for '{song_name}'...")
    result = recommend_songs(song_name, train_data)
    if result is not None:
        song_index, recommended_indices = result
        plot_recommendations(song_index, train_data, recommended_indices)
