#code 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("E:/Downloads/food.csv")

# Drop non-numeric columns for clustering
numeric_df = df.drop(columns=['Category', 'Description', 'Nutrient Data Bank Number'])

# Standardize the data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(numeric_df)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# Function to get recommendations for a specific cluster
def get_recommendations(cluster_number):
    return df[df['Cluster'] == cluster_number][['Category', 'Description']].to_dict(orient='records')

# Print recommendations for each cluster
for cluster_num in range(4):
    recommendations = get_recommendations(cluster_num)
    print(f"Recommendations for Cluster {cluster_num}:")
    for rec in recommendations[:5]:  # Display first 5 recommendations for brevity
        print(f"- {rec['Category']}: {rec['Description']}")
    print()

# Get cluster centers to understand the nutritional profile of each cluster
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=numeric_df.columns)
print("Cluster Centers (Nutritional Profile of Each Cluster):")
print(cluster_centers_df)

# Define characteristics of each cluster based on cluster centers
for i, center in enumerate(cluster_centers):
    profile = {}
    for nutrient, value in zip(numeric_df.columns, center):
        profile[nutrient] = value
    print(f"\nNutritional Profile for Cluster {i}:")
    print(profile)

    high_nutrients = [nutrient for nutrient, value in profile.items() if value > cluster_centers_df.mean(axis=0)[nutrient]]
    low_nutrients = [nutrient for nutrient, value in profile.items() if value < cluster_centers_df.mean(axis=0)[nutrient]]

    print(f"Cluster {i} tends to have higher levels of: {high_nutrients}")
    print(f"Cluster {i} tends to have lower levels of: {low_nutrients}")

# Example: Print detailed recommendations and profile for each cluster
for cluster_num in range(4):
    recommendations = get_recommendations(cluster_num)
    print(f"\nDetailed Recommendations for Cluster {cluster_num}:")
    for rec in recommendations[:5]:  # Display first 5 recommendations for brevity
        print(f"- {rec['Category']}: {rec['Description']}")
    
    # Print the nutritional profile of the cluster
    profile = cluster_centers[cluster_num]
    high_nutrients = [nutrient for nutrient, value in zip(numeric_df.columns, profile) if value > cluster_centers_df.mean(axis=0)[nutrient]]
    low_nutrients = [nutrient for nutrient, value in zip(numeric_df.columns, profile) if value < cluster_centers_df.mean(axis=0)[nutrient]]

    print(f"Nutritional Profile for Cluster {cluster_num}:")
    print(f"  Higher in: {high_nutrients}")
    print(f"  Lower in: {low_nutrients}")
