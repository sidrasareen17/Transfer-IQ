

import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

CSV_FILE = 'player_predictions.csv'


def get_processed_df():
    try:
        df = pd.read_csv(CSV_FILE)
        df.columns = df.columns.str.strip()

        analyzer = SentimentIntensityAnalyzer()

        if 'social_text' in df.columns:
            df['vader_compound_score'] = df['social_text'].apply(
                lambda x: analyzer.polarity_scores(str(x))['compound']
            )
        else:
            df['vader_compound_score'] = np.random.uniform(-0.2, 0.8, len(df))

        le = LabelEncoder()
        if 'career_stage' in df.columns:
            df['career_stage_enc'] = le.fit_transform(df['career_stage'].astype(str))
        else:
            df['career_stage_enc'] = 0

        return df

    except Exception as e:
        print("CSV ERROR:", e)
        return None


def train_models(df):
    features = [
        'goals_per90',
        'assists_per90',
        'pass_accuracy_pct',
        'vader_compound_score',
        'availability_rate',
        'career_stage_enc'
    ]

    for col in features:
        if col not in df.columns:
            df[col] = 0

    X = df[features]
    y = df['market_value_eur']

    m1 = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
    m2 = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X, y)

    return m1, m2, features


# ✅ HOME ROUTE
@app.route('/')
def home():
    return render_template('index1.html')


# ✅ PLAYERS LIST (FIXED)
@app.route('/players')
def players():
    df = get_processed_df()
    if df is None:
        return jsonify([])

    return jsonify(sorted(df['player_name'].dropna().unique().tolist()))


# ✅ PLAYER DATA + PREDICTION (FIXED + NaN FIX)
@app.route('/player/<name>')
def player(name):
    df = get_processed_df()
    if df is None:
        return jsonify({"error": "CSV load failed"})

    p_df = df[df['player_name'] == name].sort_values('season')

    if p_df.empty:
        return jsonify({"error": "No data found"})

    m1, m2, features = train_models(df)

    latest = p_df[features].iloc[-1:].values
    pred1 = m1.predict(latest)[0]
    pred2 = m2.predict(latest)[0]
    ensemble = (pred1 + pred2) / 2

    records = p_df.to_dict(orient='records')

    # ✅ ADD PREDICTIONS
    records[-1]['rf_prediction'] = float(pred1)
    records[-1]['gb_prediction'] = float(pred2)
    records[-1]['ensemble_prediction'] = float(ensemble)

    # ✅ ONLY ERROR FIX: remove NaN (IMPORTANT)
    records = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in records
    ]

    return jsonify(records)


if __name__ == '__main__':
    print("🚀 Running at http://127.0.0.1:5000")
    app.run(debug=True)
