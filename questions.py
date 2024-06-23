import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from model import LModel



def choose_k():
    inertia = []
    for k in range(1, 30):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 30), inertia, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

    silhouette_scores = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(df)
        silhouette_avg = silhouette_score(df, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plotting the Silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 20), silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True)
    plt.show()

sf = ['Bedrooms', 'Bathrooms', 'SquareFootageHouse', 'Location_Rural', 'Location_Suburban',
    'Location_Urban', 'Age', 'HeatingType_Oil', 'IsFurnished',
    'DateSinceForSale', 'KitchensQuality', 'BathroomsQuality', 'BedroomsQuality',
    'LivingRoomsQuality', 'SquareFootageGarden', 'PreviousOwnerRating', 'HeatingCosts',
    'WindowModelNames_Aluminum', 'WindowModelNames_Steel',  
     ]

odf = pd.read_csv('preprocV0.1.csv')
df = odf[odf['Price'] <= 100]
labels = df['Price']
df = df[sf]
df = df.astype(float)

binary_columns = df.columns[df.nunique() == 2]
other_columns = df.columns[~df.columns.isin(binary_columns)]

# df = df[(df['Bedrooms'] != 5) & (df['Bedrooms'] != 3)]
# df = df[df['Bathrooms'] != 3]

df.to_csv('price_100.csv', index=False)

scaler = MinMaxScaler()
df.loc[:, other_columns] = scaler.fit_transform(df[other_columns])

features = df

fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig2, ax2 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

df.boxplot(ax=ax[0])
ax[0].set_title('Overall Boxplots of Features')
ax[0].set_ylabel('Value')
ax[0].tick_params(axis='x', rotation=90, labelsize=5)

means = df.mean()
stds = df.std()
# means.plot(kind='bar', yerr=stds, ax=ax2[0])
ax2[0].errorbar(means.index, means, yerr=stds, fmt='o', ecolor='skyblue', capsize=5, capthick=2, elinewidth=2)
ax2[0].set_title('Overall Boxplots of Features')
ax2[0].set_ylabel('Value')
ax2[0].tick_params(axis='x', rotation=90, labelsize=5)

data = df.values

# choose_k()
cluster_num = 3
kmeans = KMeans(n_clusters=cluster_num)
kmeans.fit(data)

distribution = Counter(kmeans.labels_)
print(distribution)

for i in range(cluster_num):
    df[kmeans.labels_ == i].boxplot(ax=ax[i+1])
    ax[i+1].set_title('Overall Boxplots of Features')
    ax[i+1].set_ylabel('Value')
    ax[i+1].tick_params(axis='x', rotation=90, labelsize=5)

    means = df[kmeans.labels_ == i].mean()
    stds = df[kmeans.labels_ == i].std()
    # means.plot(kind='bar', yerr=stds, ax=ax2[i+1])
    ax2[i+1].errorbar(means.index, means, yerr=stds, fmt='o', ecolor='skyblue', capsize=5, capthick=2, elinewidth=2)
    ax2[i+1].set_title('Bar')
    ax2[i+1].set_ylabel('Value')
    ax2[i+1].tick_params(axis='x', rotation=90, labelsize=5)

plt.tight_layout()
# plt.show()

# Apply t-SNE
tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(features)

# Create a DataFrame with the t-SNE results
tsne_df = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
print(tsne_df.shape)
tsne_df['label'] = labels / labels.max()

plt.figure(figsize=(14, 10))
print(tsne_df['tsne1'].shape)
scatter = plt.scatter(tsne_df['tsne1'], tsne_df['tsne2'], c=kmeans.labels_, cmap='viridis')
# plt.colorbar(scatter, label='Label')
plt.title('t-SNE Plot')
plt.xlabel('tsne1')
plt.ylabel('tsne2')
# plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i in range(cluster_num):
    corr = df[kmeans.labels_ == i].corr().round(1)
    sns.heatmap(corr, ax=axes[i], cmap='coolwarm', annot=True)
    axes[i].set_title(f'Correlation Matrix for class {i}')
# plt.show()


X_train = odf[sf]

