import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load datasets dari direktori
players_data = pd.read_csv("players_data.csv")
key_stats_data = pd.read_csv("key_stats_data.csv")
attacking_data = pd.read_csv("attacking_data.csv")
attempts_data = pd.read_csv("attempts_data.csv")
distribution_data = pd.read_csv("distribution_data.csv")
defending_data = pd.read_csv("defending_data.csv")
goals_data = pd.read_csv("goals_data.csv")
goalkeeping_data = pd.read_csv("goalkeeping_data.csv")
disciplinary_data = pd.read_csv("disciplinary_data.csv")
teams_data = pd.read_csv("teams_data.csv")

# Membuat list dataset untuk ditampilkan
datasets = {
    "Players Data": players_data,
    "Key Stats Data": key_stats_data,
    "Attacking Data": attacking_data,
    "Attempts Data": attempts_data,
    "Distribution Data": distribution_data,
    "Defending Data": defending_data,
    "Goals Data": goals_data,
    "Goalkeeping Data": goalkeeping_data,
    "Disciplinary Data": disciplinary_data,
    "Teams Data": teams_data
}

# Menampilkan 5 data teratas dari setiap dataset
for name, data in datasets.items():
    print(f"\nDataset: {name}")
    print(data.head())

# Merge data based on id_player
merged_data = players_data.merge(key_stats_data, on="id_player", how="left")
merged_data = merged_data.merge(attacking_data, on="id_player", how="left")
merged_data = merged_data.merge(attempts_data, on="id_player", how="left")
merged_data = merged_data.merge(distribution_data, on="id_player", how="left")
merged_data = merged_data.merge(defending_data, on="id_player", how="left")
merged_data = merged_data.merge(goals_data, on="id_player", how="left")
merged_data = merged_data.merge(goalkeeping_data, on="id_player", how="left")
merged_data = merged_data.merge(disciplinary_data, on="id_player", how="left")

# Merge with teams data
merged_data = merged_data.merge(teams_data, on="team_id", how="left")

# Handling missing values
merged_data.fillna(0, inplace=True)

# Statistik Deskriptif
print(merged_data.describe())

# Visualisasi distribusi fitur utama
sns.pairplot(merged_data[['goals', 'assists', 'passing_accuracy(%)', 'tackles_won', 'saves', 'fouls_committed']])
plt.show()

# Agregasi data per tim
team_stats = merged_data.groupby("team").agg({
    'goals': 'mean',
    'assists': 'mean',
    'passing_accuracy(%)': 'mean',
    'tackles_won': 'mean',
    'saves': 'mean',
    'fouls_committed': 'mean',
    'ranking_matchday4': 'mean'
}).reset_index()

# Scaling data
scaler = StandardScaler()
team_stats_scaled = scaler.fit_transform(team_stats.drop(columns=['team']))

# PCA untuk visualisasi jika jumlah fitur > 2
if team_stats_scaled.shape[1] > 2:
    pca = PCA(n_components=2)
    team_stats_pca = pca.fit_transform(team_stats_scaled)
else:
    team_stats_pca = team_stats_scaled

# Hierarchical clustering
plt.figure(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(team_stats_pca, method='ward'))
plt.title("Dendrogram Hierarchical Clustering")
plt.show()

# Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=3, random_state=42)
team_stats["cluster"] = gmm.fit_predict(team_stats_pca)

# Evaluasi clustering
sil_score = silhouette_score(team_stats_pca, team_stats["cluster"])
db_score = davies_bouldin_score(team_stats_pca, team_stats["cluster"])
ch_score = calinski_harabasz_score(team_stats_pca, team_stats["cluster"])

print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")
print(f"Calinski-Harabasz Index: {ch_score:.4f}")

# Menambahkan label cluster
cluster_mapping = {0: "Tim Penyerang", 1: "Tim Bertahan", 2: "Tim Seimbang"}
team_stats["cluster_label"] = team_stats["cluster"].map(cluster_mapping)

print(team_stats[['team', 'cluster', 'cluster_label']])

# Visualisasi cluster
sns.scatterplot(x=team_stats_pca[:, 0], y=team_stats_pca[:, 1], hue=team_stats["cluster"], palette="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Cluster Tim berdasarkan Statistik")
plt.show()

import streamlit as st

# Fungsi prediksi kategori tim
def predict_team_category(team_name):
    result = team_stats[team_stats["team"] == team_name]
    if not result.empty:
        cluster_number = result["cluster"].values[0]
        cluster_name = result["cluster_label"].values[0]
        return f"{team_name} masuk ke kategori: {cluster_name} (Cluster {cluster_number})"
    else:
        return "Tim tidak ditemukan dalam data"

# Input pengguna untuk prediksi
team_name_input = st.text_input("Masukkan nama tim:")
if team_name_input:
    result = predict_team_category(team_name_input)
    st.write(result)

# Menampilkan tabel nama tim yang diurutkan berdasarkan ranking_matchday4 (dengan penggantian ranking)
sorted_teams = team_stats[['team', 'ranking_matchday4']].sort_values(by='ranking_matchday4', ascending=True)

# Ganti ranking_matchday4 menjadi ranking 1, 2, 3, dst.
sorted_teams['rank'] = range(1, len(sorted_teams) + 1)

# Menampilkan hanya nama tim dengan ranking yang baru
st.write("Daftar Tim Berdasarkan Ranking Matchday 4 (Ranking 1 hingga seterusnya):")
st.dataframe(sorted_teams[['team', 'rank']])
