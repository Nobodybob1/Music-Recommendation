from difflib import get_close_matches

def cleaning_data(df):
    # Remove duplicates
    df = df.drop_duplicates(subset=['name', 'artists']).reset_index(drop=True)

    # Remove low popularity songs
    treshold = 20
    df = df[df['popularity'] >= treshold].reset_index(drop=True)

    return df

# Fuzzy matching for track names
def find_similar_tracks(track_name, df):
    names = get_close_matches(track_name, df['name'].tolist())
    if not names:
        print("Track not found.")
        return None

    return names

# If more than one artist has the same track name, user needs to select the artist
def get_artists(track_name, df):
    matches = df[df['name'].str.lower().str.strip() == track_name.lower().strip()]
    if matches.empty:
        print("No matching track found.")
        return None
    
    artists = matches['artists'].unique()
    # if len(artists) == 1:
    #     return artists[0]
    # else:
    #     print("Multiple artists found for this track:")
    #     for i, artist in enumerate(artists):
    #         print(f"{i + 1}: {artist}")
    #     choice = int(input("Select the artist by number: ")) - 1
    #     return artists[choice] if 0 <= choice < len(artists) else None
    return artists

# Filter data
filters = {
    'explicit': True,
    'year_range': 0,
    'popularity_range': 0
}
def filter_tracks(df, filters, song_index):
    df_filtered = df.copy()

    # Filter if user doesn't want explicit content
    if not filters['explicit']:
        df_filtered = df_filtered[df_filtered['explicit'] == 0]
    
    # Filter by year range
    if filters['year_range'] > 0:
        year = df.loc[song_index, 'year']
        delta = filters['year_range']
        df_filtered = df_filtered[df_filtered['year'].between(year - delta, year + delta)]
        
    # Filter by popularity range
    if filters['popularity_range'] > 0:
        popularity = df.loc[song_index, 'popularity']
        delta = filters['popularity_range']
        df_filtered = df_filtered[df_filtered['popularity'].between(popularity - delta, popularity + delta)]

    return df_filtered.reset_index(drop=True)

