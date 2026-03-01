import pandas as pd
import numpy as np
import json
import os
from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture

app = FastAPI(title="Activity Prediction MCP Server")

LOG_FILE = "rawlogs.csv"
MATRIX_FILE = "intent_matrix.csv"
CLUSTER_MAP_FILE = "cluster_apps.json"
APP_THRESHOLD = 0.10 # Only include apps that make up >10% of the cluster intensity

# --- FUNCTION 1: THE TRAINER (Updated with Threshold) ---
def generate_cluster_app_map(gmm_means, app_features, threshold=APP_THRESHOLD):
    cluster_map = {}
    for i, intensities in enumerate(gmm_means):
        # Filter: Only keep apps where the intensity is > threshold
        relevant_apps = [
            app_features[idx] for idx, val in enumerate(intensities) if val >= threshold
        ]
        cluster_map[f"Cluster_{i+1}"] = relevant_apps
    return cluster_map

@app.post("/train")
def train_activity_model():
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="usagelogs.csv not found.")

    df = pd.read_csv(LOG_FILE, parse_dates=['timestamp'])
    df['diff'] = df['timestamp'].diff().dt.total_seconds() / 60
    df['session_id'] = ((df['diff'] > 20) | (df['diff'].isna())).cumsum()
    sessions = df.groupby('session_id')['app'].apply(lambda x: " ".join(x)).reset_index()

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sessions['app']).toarray()
    X_norm = X / (X.sum(axis=1)[:, np.newaxis] + 1e-9)

    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_norm)
    
    apps = vectorizer.get_feature_names_out()
    
    # Save P(Cluster | App) Matrix
    df_means = pd.DataFrame(gmm.means_, columns=apps)
    matrix = (df_means / df_means.sum(axis=0)).T
    matrix.columns = [f"Cluster_{i+1}" for i in range(3)]
    matrix.to_csv(MATRIX_FILE)

    # Save Cluster Profiles based on threshold
    cluster_map = generate_cluster_app_map(gmm.means_, apps)
    with open(CLUSTER_MAP_FILE, 'w') as f:
        json.dump(cluster_map, f)
    
    return {"message": "Model trained.", "cluster_profiles": cluster_map}

# --- FUNCTION 2: THE PREDICTOR ---
@app.get("/predict/{app_name}")
def get_app_prediction(app_name: str):
    if not os.path.exists(MATRIX_FILE) or not os.path.exists(CLUSTER_MAP_FILE):
        raise HTTPException(status_code=400, detail="Model not trained.")
    
    matrix = pd.read_csv(MATRIX_FILE, index_col=0)
    with open(CLUSTER_MAP_FILE, 'r') as f:
        cluster_map = json.load(f)

    app_query = app_name.lower()
    if app_query not in matrix.index:
        return {"app": app_query, "is_ambiguous": True, "reason": "Unknown App"}

    probs = matrix.loc[app_query].to_dict()
    top_cluster = max(probs, key=probs.get)
    
    # If top prob < 0.80, we consider it ambiguous
    is_ambiguous = probs[top_cluster] < 0.80

    # Dynamic Suggestion Logic
    if is_ambiguous:
        # Return apps for ALL clusters where the current app has > 15% presence
        # This gives the LLM the 'menu' of choices to narrow down
        relevant_clusters = {c: cluster_map[c] for c, p in probs.items() if p > 0.15}
    else:
        # Return only the top cluster apps
        relevant_clusters = {top_cluster: cluster_map[top_cluster]}

    return {
        "app": app_query,
        "is_ambiguous": is_ambiguous,
        "probabilities": probs,
        "contextual_clusters": relevant_clusters
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)