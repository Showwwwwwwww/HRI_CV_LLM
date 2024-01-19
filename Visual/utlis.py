import numpy as np
from numpy.linalg import norm





def feature_compare(feature1, feature2, threshold):
    """
    Method 1:
        # Euclidean distance to compare with threshold
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        if dist < threshold:
            return True
        else:
            return False
    """
    # Method 2 - Cosine Similarity
    cosine = np.dot(feature1, feature2.T) / (norm(feature1) * norm(feature2))
    if cosine > 0.7:
        return True
    return False