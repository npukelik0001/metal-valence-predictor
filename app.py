import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load cleaned dataset
df = pd.read_csv("df_cleaned.csv")

# Recover missing columns if needed
if 'track_genre' not in df.columns and 'genre_for_plot' in df.columns:
    df['track_genre'] = df['genre_for_plot']
if 'artists' not in df.columns:
    df['artists'] = "Unknown"

# Page config
st.set_page_config(page_title="Metal Valence Predictor", layout="wide")

# Disclaimer
st.markdown("_Disclaimer: Due to many external factors influencing this project, the R-squared value is only about 51%._")

# Custom CSS for blue background and white text everywhere, with good slider and tab styles
st.markdown(
    """
    <style>
    /* Background and container */
    body, .block-container {
        background-color: #1a3c72 !important;  /* medium blue */
        color: white !important;
    }

    /* Title white */
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        color: white !important;
        padding: 10px 0px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sub-header white */
    .sub-header {
        font-size: 1.2rem;
        color: white !important;
        margin-bottom: 30px;
    }

    /* Sliders styling */
    div[data-baseweb="slider"] > div > div {
        background-color: #255aaf !important; /* darker blue track */
    }
    div[data-baseweb="slider"] > div > div > div {
        background-color: #80b3ff !important; /* lighter blue filled */
    }
    div[data-baseweb="slider"] > div > div > div > div {
        background-color: white !important; /* white handle */
        border: 2px solid white !important;
    }

    /* Tabs label text color white */
    .css-1r6slb0.e1fqkh3o3 > button {
        color: white !important;
    }
    /* Tab selected underline white */
    .css-1r6slb0.e1fqkh3o3 > button[aria-selected="true"] {
        border-bottom: 3px solid white !important;
        font-weight: 700;
    }

    /* DataFrame background and text */
    .stDataFrame {
        background-color: #27496d !important; /* dark blue-gray */
        color: white !important;
    }

    /* Fix column text color in tab1 sliders and other input controls */
    .stSlider > div, .stSlider > div > div {
        color: white !important;
    }

    /* Ensure text inputs, selectboxes, buttons text are white */
    div[role="combobox"], div[role="textbox"], button {
        color: white !important;
    }

    /* Matplotlib text in plots white */
    .stPlotlyChart, .stPyplot {
        color: white !important;
    }

    /* Set matplotlib axes labels and titles color */
    .mpl-color {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="main-title">Metal Song Valence Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Estimate the emotional positivity of metal tracks using audio features</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict Valence", "Top Songs by Genre", "Top Songs by Artist"])

# --- TAB 1: Predict Valence ---
with tab1:
    st.markdown("## Predict Valence from Audio Features")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
    with col2:
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -5.0)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    with col3:
        duration_min = st.slider("Duration (minutes)", 0.5, 15.0, 4.0)

    input_dict = {
        'danceability': danceability,
        'energy': energy,
        'loudness': loudness,
        'speechiness': speechiness,
        'duration_min': duration_min
    }

    sample = pd.DataFrame([input_dict])
    sample = sample.reindex(columns=scaler.feature_names_in_, fill_value=0)
    input_scaled = scaler.transform(sample)
    input_scaled_df = pd.DataFrame(input_scaled, columns=scaler.feature_names_in_)
    input_scaled_df = input_scaled_df.drop(columns=['duration_ms'], errors='ignore')

    valence_pred = model.predict(input_scaled_df)[0]

    st.subheader("Predicted Valence Score")
    st.metric(label="Valence (0 = sad, 1 = happy)", value=f"{valence_pred:.3f}")

    mood = "Dark / Sad" if valence_pred < 0.3 else "Neutral / Mixed" if valence_pred < 0.6 else "Uplifting / Positive"
    st.write("Mood Interpretation:", mood)

    st.markdown("### Songs Most Similar to Your Input")
    similar_df = df.copy().drop_duplicates(subset='track_name')

    drop_cols = ['valence', 'artists', 'track_name', 'album_name', 'tempo_category_plot', 'genre_for_plot', 'key_label']
    X_all = similar_df.drop(columns=drop_cols, errors='ignore')
    X_all = X_all.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_all_scaled = scaler.transform(X_all)
    X_all_scaled = pd.DataFrame(X_all_scaled, columns=scaler.feature_names_in_)
    X_all_scaled = X_all_scaled.drop(columns=['duration_ms'], errors='ignore')

    distances = euclidean_distances(X_all_scaled, input_scaled_df)
    similar_df['Distance'] = distances.flatten()

    closest_matches = similar_df[['track_name', 'artists', 'valence', 'Distance']].sort_values(by='Distance').head(10)
    closest_matches.index = np.arange(1, len(closest_matches) + 1)
    st.dataframe(closest_matches.style.format({'valence': '{:.3f}', 'Distance': '{:.3f}'}))

# --- TAB 2: Genre Songs + Graphs ---
with tab2:
    st.markdown("## Top 10 Songs by Metal Subgenre")
    if 'track_genre' in df.columns:
        genre_list = sorted(df['track_genre'].dropna().unique())
        selected_genre = st.selectbox("Choose a metal subgenre:", genre_list)

        genre_df = df[df['track_genre'] == selected_genre].copy().drop_duplicates(subset='track_name')

        drop_cols_genre = ['artists', 'track_name', 'album_name', 'valence', 'tempo_category_plot', 'genre_for_plot', 'key_label']
        X_genre = genre_df.drop(columns=drop_cols_genre, errors='ignore')
        X_genre = X_genre.reindex(columns=scaler.feature_names_in_, fill_value=0)
        X_genre_scaled = scaler.transform(X_genre)
        X_genre_scaled = pd.DataFrame(X_genre_scaled, columns=scaler.feature_names_in_)
        X_genre_scaled = X_genre_scaled.drop(columns=['duration_ms'], errors='ignore')

        genre_df['Predicted_Valence'] = model.predict(X_genre_scaled)
        top10 = genre_df[['track_name', 'artists', 'Predicted_Valence']].sort_values(by='Predicted_Valence', ascending=False).head(10)
        top10.index = np.arange(1, len(top10) + 1)
        st.dataframe(top10.style.format({'Predicted_Valence': '{:.3f}'}))

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(genre_df['Predicted_Valence'], bins=20, color='white', edgecolor='black')
            ax.set_title(f"Valence Distribution for {selected_genre}", color='white')
            ax.set_xlabel("Predicted Valence", color='white')
            ax.set_ylabel("Frequency", color='white')
            st.pyplot(fig)

        with col2:
            if 'energy' in genre_df.columns:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.scatter(genre_df['energy'], genre_df['Predicted_Valence'], alpha=0.6, color='white')
                ax2.set_xlabel("Energy", color='white')
                ax2.set_ylabel("Predicted Valence", color='white')
                ax2.set_title(f"Energy vs Valence for {selected_genre}", color='white')
                st.pyplot(fig2)
    else:
        st.warning("Genre column not found in dataset.")

# --- TAB 3: Artist Songs + Graphs ---
with tab3:
    st.markdown("## Top Songs by Selected Artist")

    artist_counts = df['artists'].value_counts()
    filtered_artists = artist_counts[artist_counts >= 5].index.tolist()
    available_artists = sorted(filtered_artists)

    if available_artists:
        selected_artist = st.selectbox("Choose an artist:", available_artists)

        artist_df = df[df['artists'] == selected_artist].copy().drop_duplicates(subset='track_name')

        drop_cols_artist = ['artists', 'track_name', 'album_name', 'valence', 'tempo_category_plot', 'genre_for_plot', 'key_label']
        X_artist = artist_df.drop(columns=drop_cols_artist, errors='ignore')
        X_artist = X_artist.reindex(columns=scaler.feature_names_in_, fill_value=0)
        X_artist_scaled = scaler.transform(X_artist)
        X_artist_scaled = pd.DataFrame(X_artist_scaled, columns=scaler.feature_names_in_)
        X_artist_scaled = X_artist_scaled.drop(columns=['duration_ms'], errors='ignore')

        artist_df['Predicted_Valence'] = model.predict(X_artist_scaled)
        top_artist_songs = artist_df[['track_name', 'Predicted_Valence']].sort_values(by='Predicted_Valence', ascending=False).head(10)
        top_artist_songs.index = np.arange(1, len(top_artist_songs) + 1)
        st.dataframe(top_artist_songs.style.format({'Predicted_Valence': '{:.3f}'}))

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            ax1.barh(top_artist_songs['track_name'], top_artist_songs['Predicted_Valence'], color='white')
            ax1.set_xlabel("Predicted Valence", color='white')
            ax1.set_title(f"Top 10 Songs by {selected_artist}", color='white')
            ax1.invert_yaxis()
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.boxplot(artist_df['Predicted_Valence'], patch_artist=True, boxprops=dict(facecolor='white'))
            ax2.set_title(f"Valence Distribution for {selected_artist}", color='white')
            ax2.set_ylabel("Predicted Valence", color='white')
            st.pyplot(fig2)
    else:
        st.warning("No artists with at least 5 songs found in the dataset.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit · Metal Genre Audio AI · By Nastassia Pukelik")
