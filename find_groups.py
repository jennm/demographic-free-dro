import numpy as np

from get_embeddings import create_dataloader
from sklearn.linear_model import LogisticRegression

def visualize_group(group_info):
    pass

def relabel_points(X, y):
    pass

def train_classifier(X, y, k=1):
    model = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', max_iter=500)

    for _ in range(k):
        model.fit(X, y)
        relabel_points(model, y)

    return model

def subsample(predictions, embeddings, subclasses, labels, data_idx):
    # uniform random one thirds of the points
    num_points = predictions.shape[0]
    num_samples = num_points // 3
    return np.random.choice(num_points, num_samples, replace=False)

def unpack_data():
    train_npz = np.load('cmnist_meta_train.npz')
    predictions = train_npz['predictions']
    embeddings = train_npz['embeddings']
    subclasses = train_npz['subclass']
    labels = train_npz['label']
    data_idx = train_npz['data_idx']
    
    return predictions, embeddings, subclasses, labels, data_idx

def find_groups():
    # load data
    predictions, embeddings, subclasses, labels, data_idx = unpack_data()
    misclassified = np.argmax(predictions, axis=1) != labels

    # setup groups
    # TODO: 
    groups = {0 : np.array([0] * predictions.shape[0])}

    # subsample
    subsample_indices = subsample(predictions, embeddings, subclasses, labels, data_idx)

    curr_group_id = len(groups)

    # while criteria
    while curr_group_id < 3 and np.sum(misclassified[subsample_indices]) > 0:
        # train classifier
        trained_model = train_classifier(embeddings[subsample_indices], misclassified[subsample_indices])
        learned_group = trained_model.predict(embeddings)

        # NOTE: maybe should just retain true positives i.e. only actually misclassified points 
        # => this means that we only discover subgroups among the misclassified population
        # => this means we aren't introducing potentially incorrect group info for points the model is already getting correct
        # => this sounds like a good way to control for performance

        # delete discovered group from subsample
        subsample_learned_group = learned_group[subsample_indices]

        ##############################
        true_positives = subsample_learned_group[misclassified[subsample_indices]]
        print('# True Positives:', np.sum(true_positives))
        print('# Total Misclassified in Subsample:', np.sum(misclassified[subsample_indices]))
        print('Learned Subsample Group Size:', np.sum(subsample_learned_group))
        print('Learned Trainset Group Size:', np.sum(learned_group))
        ##############################

        learned_group_mask = ~subsample_learned_group
        subsample_indices = subsample_indices[learned_group_mask]

        # mark group
        learned_group = np.where(learned_group, curr_group_id, -1)
        groups[curr_group_id] = learned_group        

        # visualize identified group
        visualize_group(learned_group)

        curr_group_id += 1
    
    print('# Learned Groups:', len(groups.keys()))
    print('Learned Group Ids:', groups.keys())

    # write classifier group file
    save_idx = list(groups.keys())

    save_idx.sort()
    store_groups = [np.array(groups[i]) for i in save_idx]
    stacked_groups = np.stack(store_groups)
    np.savez('classifier_groups.npz', group_array=stacked_groups.T)

















def __find_groups(train_data, val_data, feature_extractor, **loader_kwargs):
    train_loader = create_dataloader(feature_extractor, train_data, None, loader_kwargs)
    val_loader = create_dataloader(feature_extractor, val_data, None, loader_kwargs)

    experiment(train_loader, 'train')
    experiment(val_loader, 'val')

def experiment(data_loader, desc):
    store_pred = []
    store_emb = []
    store_sub = []
    store_label = []
    store_data_idx = []
    store_loss = []

    for batch in data_loader:
        predictions = batch['embeddings'][str(0)]
        store_pred.append(predictions)

        embedding = batch['embeddings'][str(1)]
        store_emb.append(embedding)

        true_subclass = batch['group']
        store_sub.append(true_subclass)

        true_label = batch['labels']
        store_label.append(true_label)

        loss = batch['loss']
        store_loss.append(loss)

        data_idx = batch['idx']
        store_data_idx.append(data_idx)
    
    store_pred = np.concatenate(store_pred)
    store_emb = np.concatenate(store_emb)
    store_sub = np.concatenate(store_sub)
    store_label = np.concatenate(store_label)
    store_loss = np.concatenate(store_loss)
    store_data_idx = np.concatenate(store_data_idx)

    print((np.argmax(store_pred, axis=1) == store_label).sum() / len(store_label))

    misclassified = np.argmax(store_pred, axis=1) != store_label
    print(misclassified.sum())

    print(store_label.sum())
    print(np.argmax(store_pred, axis=1).sum())

    np.savez(f'cmnist_meta_{desc}.npz', predictions=store_pred, embeddings=store_emb, subclass=store_sub, label=store_label, loss=store_loss, data_idx=store_data_idx)
