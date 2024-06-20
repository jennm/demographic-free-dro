import numpy as np
from collections import Counter
from scipy.spatial import distance


def unpack_data():
    train_npz = np.load('cmnist_meta_train.npz')
    predictions = train_npz['predictions']
    embeddings = train_npz['embeddings']
    subclasses = train_npz['subclass']
    labels = train_npz['label']
    data_idx = train_npz['data_idx']
    
    return predictions, embeddings, subclasses, labels, data_idx

def breakdown():
    # load data
    predictions, embeddings, subclasses, labels, data_idx = unpack_data()
    misclassified = np.argmax(predictions, axis=1) != labels
    # embeddings = np.hstack((embeddings, misclassified.reshape(-1, 1)))

    # reducer = umap.UMAP()
    # reduced_embeddings = reducer.fit_transform(embeddings)

    # misclassified_indices = np.where(misclassified)[0]
    # minor_indices = np.where(subclasses == 4)[0]

    embeddings = embeddings[misclassified]
    subclasses = subclasses[misclassified]

    # print('len options', len(embeddings))
    # print('num misclassified', misclassified.sum())

    # random_index = np.random.randint(0, len(embeddings))
    print('# Misclassified 19s:', np.where((subclasses == 19))[0])
    random_index = np.random.choice(np.where((subclasses == 19))[0])
    print('Random Subclass', subclasses[random_index])
    random_embedding = embeddings[random_index]

    # embeddings = embeddings[misclassified]
    # subclasses = subclasses[misclassified]

    distance_metric = 'cosine'
    distances = distance.cdist([random_embedding], embeddings, distance_metric).flatten()
    nearest_indices = np.argsort(distances)[1:101]  # Exclude the point itself
    nearest_labels = subclasses[nearest_indices]

    label_counts = Counter(nearest_labels)
    most_common_labels = label_counts.most_common(5)

    print("Top 5 most common labels among nearest embeddings:")
    for label, count in most_common_labels:
        print(f"Label {label}: {count} occurrences")