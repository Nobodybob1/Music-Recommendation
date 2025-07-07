# Music Recommendation System

This project is a simple, interactive **music recommendation system** built using Python, Streamlit and scikit-learn. Users can enter a song name, select correct artist, and receive **5 similar songs** based on audio features like energy, valence, danceability and more.

The system uses a **K-Nearest Neighbors (KNN)** model with **cosine similarity** and allows filtering by:
- Explicit content
- Year range
- Populariy range

It also includes a 2D **PCA visualization** showing where the input song and recommendations appear in feature space.



## Features
- Fuzzy search for song title
- Choosing artist (if there is more than one)
- Optional filtering
- KNN recommendation using audio features
- PCA-based plot of similar songs
- Streamlit interface



## Project structure
- app.py # Main Streamlit app
- helper_functions.py # Cleaning data, Fuzzy searching, getting artists, filtering
- main.ipynb # Notebook with visualizing data and general logic
- requirements.txt - # Requirements



## Running the app
Run: **streamlit run app.py**



## Example Workflow
1. Type `hurt` in the input box
2. Select `Hurt` from songs
3. Select `Nine Inch Nails` from artists
4. Adjust filters
5. See a list of similar songs + PCA plot



## Possible Improvements
* Use real Spotify API for artist genre and audio analysis
* Cluster songs by style (e.g. with KMeans)
* Deploy to Streamlit Cloud


## Link to Dataset:
**https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset**