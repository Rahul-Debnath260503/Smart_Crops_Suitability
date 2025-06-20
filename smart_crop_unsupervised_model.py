# -*- coding: utf-8 -*-
"""Smart_crop_unsupervised_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1d_gmLvBnAYFRKPllCM8bqYmF2U4Zy6gK
"""

#  Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

#  Step 2: Load Dataset
df = pd.read_csv("/content/drive/MyDrive/final_project_AGP/smart_crop_lat_long.csv")
df.head()

#  Step 3: Select and Clean Relevant Features
features = [
    "NDVI",
    "Seasonal_Rainfall",
    "Annual_Rainfall",
    "Temperature",
    "pH_0_5",
    "Carbon",
    "Texture"
]

df_clean = df[features].dropna()

#  Step 4: Scale the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Step 5: Determine Optimal Number of Clusters (Elbow Method)
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='s', color='green')
plt.title("Silhouette Score")
plt.xlabel("No. of Clusters")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.show()

#  Step 6: Apply KMeans with Chosen K (e.g., 4)
optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_clustered = df_clean.copy()
df_clustered["Cluster"] = clusters

#  Step 7: Analyze Cluster Profiles
cluster_summary = df_clustered.groupby("Cluster").mean()
cluster_summary.style.background_gradient(cmap='YlGnBu')

#  Step 8: Visualize with PCA (2D projection)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df_clustered["PCA1"] = pca_result[:, 0]
df_clustered["PCA2"] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clustered, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=60)
plt.title("PCA: Cluster Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

#  Updated: 5-cluster crop recommendation logic
cluster_labels = {
    0: "Cluster 0: Apple, Wheat, Barley, Maize, Groundnut, Pearl Millet, Mushroom, Ginger (Low Rainfall, Alkaline Soil)",
    1: "Cluster 1: Maize (Corn),Rice (Paddy), Tomatoes, Pineapple, Groundnut (Peanut), Cotton, Sugarcane (High Temperature, Moderate Rainfall)",
    2: "Cluster 2: Paddy, Maize , Soybean, Sweet Potato, Cotton, Sugarcane, Wheat (Moderate Rainfall, High NDVI, Varying pH)",
    3: "Cluster 3: Apple, Oats,Carrot,Spinach,Radish,Blueberries,Strawberries, Barley, Potato (Cool Climate, Moderate Rainfall, Acidic Soil)",
    4: "Cluster 4: Coconut, Rubber, Banana, Cocoa, Black Pepper, Oil Palm (Very High Rainfall, Acidic Soil, High Temperature)",
    5: "Cluster 5: Coconut, Banana, Black Pepper, Taro, Cocoa(Hot Climate, Acidic Soil, Heavy Rainfall)",

}

# Apply to dataframe
df_clustered["Recommended_Crops"] = df_clustered["Cluster"].map(cluster_labels)

# Show result
df_clustered[["PCA1", "PCA2", "Cluster", "Recommended_Crops"]].head(20)

df_clustered.value_counts("Cluster")

# Plot average values per cluster
import matplotlib.pyplot as plt

cluster_features = df_clustered.groupby("Cluster").mean(numeric_only=True)
cluster_features.drop(columns=["PCA1", "PCA2"], inplace=True)  # Remove PCA cols

cluster_features.T.plot(kind="bar", figsize=(12, 6))
plt.title("Average Feature Values per Cluster")
plt.ylabel("Scaled Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

df_clustered.to_csv("clustered_crop_recommendations.csv", index=False)

import folium
import pandas as pd # Ensure pandas is imported if this is a separate cell

# Create a map centered on your data
# map_clusters = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=6) # Original line

# Instead of starting with df, filter the original df based on the cleaned rows
# Get the index of the cleaned rows
cleaned_index = df_clean.index

# Use this index to select the corresponding rows from the original df, including Latitude and Longitude
df_clustered = df.loc[cleaned_index].copy()

# Add the cluster results to this new df_clustered
df_clustered["Cluster"] = clusters
# Add the PCA results as well (if needed for other visualizations/analysis)
df_clustered["PCA1"] = pca_result[:, 0]
df_clustered["PCA2"] = pca_result[:, 1]

# Add the Recommended_Crops column using the updated cluster_labels
cluster_labels = {
    0: "Cluster 0: Apple, Wheat, Barley, Maize, Groundnut, Pearl Millet, Mushroom, Ginger (Low Rainfall, Alkaline Soil)",
    1: "Cluster 1: Maize (Corn),Rice (Paddy), Tomatoes, Pineapple, Groundnut (Peanut), Cotton, Sugarcane (High Temperature, Moderate Rainfall)",
    2: "Cluster 2: Paddy, Maize , Soybean, Sweet Potato, Cotton, Sugarcane, Wheat (Moderate Rainfall, High NDVI, Varying pH)",
    3: "Cluster 3: Apple, Oats,Carrot,Spinach,Tea,Radish,Blueberries,Strawberries, Barley, Potato (Cool Climate, Moderate Rainfall, Acidic Soil)",
    4: "Cluster 4: Coconut, Rubber, Oil Palm, Durian, Rambutan (Very High Rainfall >3000mm, Acidic Soil pH 4.5-5.5, High Temperature 28-35°C)",

    5: "Cluster 5: wheat, barley, maize, millets, pulses, oilseeds, (Hot Climate 25-32°C, Acidic Soil pH 5.0-6.0, Heavy Rainfall)"
}
df_clustered["Recommended_Crops"] = df_clustered["Cluster"].map(cluster_labels)

# Now df_clustered contains 'Latitude', 'Longitude', 'Cluster', and 'Recommended_Crops' for the clustered rows

# Create the map using the coordinates from df_clustered
map_clusters = folium.Map(location=[df_clustered["Latitude"].mean(), df_clustered["Longitude"].mean()], zoom_start=6)


# Color by cluster - Using colors extracted from Matplotlib plot
cluster_colors = {
    0: '#66c2a5',  # Teal green
    1: '#8da0cb',  # Soft blue
    2: '#a6d854',  # Lime green
    3: '#e5c494',  # Tan
    4: '#b3b3b3',  # Light gray
    5: '#fc8d62',  # Coral orange
    
}

for index, row in df_clustered.iterrows():
    # Ensure the color list has enough elements for the number of clusters
    cluster_index = int(row["Cluster"]) # Ensure cluster is integer for indexing
    if cluster_index < len(cluster_colors):
        marker_color = cluster_colors[cluster_index]
    else:
        # Provide a default color or handle the error if more clusters than colors
        marker_color = "gray" # Default color

    # Create popup content
    popup_content = f"<b>Cluster:</b> {row['Cluster']}<br><b>Recommended Crops:</b> {row['Recommended_Crops']}"

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=3,
        color=marker_color,
        fill=True,
        fill_opacity=0.4,
        popup=popup_content # Add the popup content here
    ).add_to(map_clusters)

map_clusters  # Displays the map

# Analyze the distribution of states within each cluster
cluster_state_distribution = df_clustered.groupby("Cluster")["State"].value_counts().unstack(fill_value=0)

# Display the distribution
print("Distribution of States within each Cluster:")
display(cluster_state_distribution)

# Optional: Visualize the distribution
cluster_state_distribution.T.plot(kind='bar', stacked=True, figsize=(12, 7))
plt.title("Distribution of States per Cluster")
plt.xlabel("State")
plt.ylabel("Number of Samples")
plt.xticks(rotation=90)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
