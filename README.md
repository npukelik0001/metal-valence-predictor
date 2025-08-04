# Metal Song Emotional Valence Predictor 🎸

This project applies regression modeling to predict the emotional positivity (valence) of metal songs based on their audio features. It was developed for CIS 9660 - Data Mining for Business Analytics and deployed using Streamlit.

---

## Overview

The goal of this project is to help independent artists and music teams in the metal scene identify which songs are more emotionally positive and potentially more appealing for promotion. The target variable is **valence**, a score from 0 (sad) to 1 (happy), pulled from Spotify's audio analysis API.

---

## Dataset

- **Source**: [Spotify Tracks Dataset by Maharshi Pandya (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data)
- **Filtered Size**: ~4,000 songs from metal subgenres
- **Genres used**: black-metal, death-metal, heavy-metal, metalcore, metal
- **Key Features**: danceability, energy, loudness, speechiness, duration, tempo, key, and more

---

## Data Preprocessing

- Dropped missing/duplicate rows
- Filtered for metal-related subgenres
- One-hot encoded tempo categories, subgenres, and keys
- Scaled numerical features using StandardScaler

---

## Model Development

Trained and compared the following regression models:

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest  
- Gradient Boosting  
- **XGBoost** ✅ (Best performer)

**Metrics used**: R², RMSE, MAE  
**Final model**: XGBoost — saved as `xgb_model.pkl`  
**Scaler**: StandardScaler — saved as `scaler.pkl`

---

## Streamlit App Features

- Predict emotional valence based on user input
- Mood feedback (sad, neutral, or uplifting)
- Similar song recommendations by distance
- Visualizations:
  - Valence distribution
  - Energy vs Valence scatter plot
  - Top tracks by genre and artist

Note: Artists shown must have at least 5 songs. Duplicates removed from track recommendations.

---

## Repository Contents

- `app.py` — Streamlit web application  
- `spotify[emotional_valence].py` — All modeling and evaluation  
- `df_cleaned.csv` — Cleaned dataset  
- `xgb_model.pkl` — Trained model  
- `scaler.pkl` — Feature scaler  
- `requirements.txt` — Python packages  
- `README.md` — You’re reading it  
- `Nastassia_Pukelik_Technical_Report.pdf` — Technical summary report  

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. **Navigate to the project folder**
   ```bash
   cd your-repo-name
   ```

3. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Kaggle API**
   Create a `.env` file with:
   ```env
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_key
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## Disclaimer

This model explains about **51% of the variation in valence**, which is reasonable given that musical emotion is influenced by many subjective and unmeasurable factors. This was built for educational use only.

---

## Author

**Nastassia Pukelik**  
Baruch College – CIS 9660  
Summer 2025