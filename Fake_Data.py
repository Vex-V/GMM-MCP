import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. DATA GENERATION (Unlabeled/Unsupervised) ---
def generate_unlabeled_data(filename="rawlogs.csv", num_events=600):
    # Hidden habits the model has to discover
    habits = [
        ["vscode", "chrome", "terminal", "docker"], # Cluster 1: Dev
        ["spotify", "discord", "chrome", "steam"],  # Cluster 2: Play
        ["outlook", "slack", "chrome", "excel"],    # Cluster 3: Work
    ]
    data = []
    current_time = datetime.now()
    
    for _ in range(150): # Generate 150 sessions
        current_habit = random.choice(habits)
        session_apps = random.sample(current_habit, random.randint(2, 4))
        for app in session_apps:
            current_time += timedelta(minutes=random.randint(1, 10))
            data.append({"timestamp": current_time, "app": app})
        current_time += timedelta(hours=random.randint(1, 4)) # Gap between sessions

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df

generate_unlabeled_data()