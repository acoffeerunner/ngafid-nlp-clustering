from umap import UMAP
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation

import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 21
np.random.seed(RANDOM_SEED)


df = pd.read_csv('../data/preprocessed_new_corenlp_grouped.csv')
df['cleaned_problem'] = df.fillna(value=' ')['cleaned_problem'] 

# preparing text for vectorization
all_text = df['cleaned_problem'].values
all_text = [' '.join(word for word in text.replace('service bulletin', '').replace('ad e ', '').replace('ad ', '').replace('maintenance flight require', '').replace('service bulletin', '').replace('serial number', '').split() 
                           if word not in ('forward', 'left', 'right', 'rear', 'upper', 'lower', 'large', 'small', 'insp', 'change', 'time'))
                  for text in all_text]
all_text = [' '.join('intake' if word == 'intakes' else word for word in text.split()) for text in all_text]


# vectorization
embedder = TfidfVectorizer(use_idf=True, norm='l2', sublinear_tf=False, max_df=0.99, min_df=0.0)
corpus_embeddings = embedder.fit_transform(all_text)

# dimensionality reduction
reducer = UMAP(n_neighbors=50, min_dist=0.0, n_components=50, random_state=42, unique=True, metric='cosine')
reduced = reducer.fit_transform(corpus_embeddings)

# clusterer
clusterer = HDBSCAN(cluster_selection_epsilon=0.105, cluster_selection_method='leaf', leaf_size=1, min_samples=20, min_cluster_size=5, metric='euclidean')
df['n_cluster'] = clusterer.fit_predict(reduced)
print(f"Clusters formed: {np.unique(df['n_cluster'].values).shape[-1] - 1}")
print(f"Noise points: {np.where(df['n_cluster'].values == -1)[-1].shape[-1]}")


# noise points integration

# 1. Build a mapping from merged_cluster → list of doc indices
clusters = {}
n_clusters = df[df['n_cluster'] > -1]['n_cluster'].values
for doc_idx, cl in enumerate(n_clusters):
    if cl >= 0:
        clusters.setdefault(cl, []).append(doc_idx)

# 2. Compute medoid for each merged cluster
medoids = {}
for label in range(n_clusters.max() + 1):
    idx = df[df['n_cluster'] == label].index
    if len(idx) == 0:
        continue
    cluster_points = reduced[idx]
    sim = cosine_similarity(cluster_points)
    avg_sim = sim.mean(axis=1)
    local_medoid_idx = np.argmax(avg_sim)
    medoid = cluster_points[local_medoid_idx]
    medoids[label] = medoid
    
# Stack medoids into matrix for fast lookup
cluster_ids = sorted(medoids)
M = np.vstack([medoids[cl] for cl in cluster_ids])  # shape=(n_clusters, n_feats)

# 3. For each noise point, find best‐matching cluster
noise_indices = [i for i, lbl in enumerate(df['n_cluster'].values) if lbl == -1]
noise_mat = reduced[noise_indices]              # shape=(n_noise, n_feats)
# cosine_similarity → shape (n_noise, n_clusters)
sim_to_clusters = cosine_similarity(noise_mat, M)

# 4. Assign if max similarity ≥ noise_threshold (e.g. 0.75)
noise_threshold = 0.999925
best_sim = sim_to_clusters.max(axis=1)
best_cl_idx = sim_to_clusters.argmax(axis=1)

# Build a copy of final_labels
assigned_labels = df['n_cluster'].values.copy()
for ni, doc_idx in enumerate(noise_indices):
    if best_sim[ni] >= noise_threshold:
        assigned_labels[doc_idx] = cluster_ids[best_cl_idx[ni]]

df['cluster'] = assigned_labels

print(f"Noise points after noise integration: {np.where(df['cluster'].values == -1)[-1].shape[-1]}")

df = df.drop('n_cluster', axis = 1)

# Perform LDA for topic modeling
lda = LatentDirichletAllocation(n_components=10, random_state=RANDOM_SEED)
lda_topics = lda.fit_transform(corpus_embeddings)

# Get feature names (words) from TF-IDF vectorizer
feature_names = embedder.get_feature_names_out()

# Assign meaningful topic labels to each cluster
topic_labels = []
for cluster_id in sorted(df['cluster'].unique()):
    if cluster_id == -1:
        continue
        
    cluster_indices = df[df['cluster'] == cluster_id].index
    cluster_embeddings = corpus_embeddings[cluster_indices]
    
    # Get top words for this cluster
    cluster_texts = df.iloc[cluster_indices]['cleaned_problem']
    cluster_vectorizer = TfidfVectorizer(max_features=5)
    cluster_tfidf = cluster_vectorizer.fit_transform(cluster_texts)
    
    # Get top 5 keywords for the cluster
    feature_array = np.array(cluster_vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(cluster_tfidf.toarray().mean(axis=0)).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:5]
    
    topic_labels.append(f"{', '.join(top_keywords)}")

# Map topic labels to clusters
df['topic_label'] = df['cluster'].map(lambda x: topic_labels[x])

df_groups = df[df['cluster'] > -1].groupby(by = 'cluster')

if not os.path.exists('../data/cluster_information'):
    os.makedirs('../data/cluster_information')

for cluster, group in df_groups:
    group.to_csv(f'../data/cluster_information/c_{cluster}.csv', index=False)

print(f"Silhouette score: {silhouette_score(corpus_embeddings[df['cluster'] > -1], df[df['cluster'] > -1]['cluster']): .4f}")