import numpy as np
from scipy.stats import pearsonr  # Pearson correlation
from scipy.spatial.distance import cosine, euclidean, cityblock

# Define Jaccard similarity function
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Sample data
data1 = np.array([1, 2, 3, 4, 5])
data2 = np.array([2, 4, 6, 8, 10])

# Sets for Jaccard similarity
set1 = set([1, 2, 3])
set2 = set([2, 3, 4, 5])

# Pearson Correlation
pearson_corr, _ = pearsonr(data1, data2)
print('Pearson Correlation:', pearson_corr)

# Cosine Similarity
cos_sim = 1 - cosine(data1, data2)
print('Cosine Similarity:', cos_sim)

# Jaccard Similarity
jaccard_sim = jaccard_similarity(set1, set2)
print('Jaccard Similarity:', jaccard_sim)

# Euclidean Distance
euclidean_dist = euclidean(data1, data2)
print('Euclidean Distance:', euclidean_dist)

# Manhattan (Cityblock) Distance
manhattan_dist = cityblock(data1, data2)
print('Manhattan Distance:', manhattan_dist)
