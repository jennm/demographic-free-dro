import numpy as np

from get_embeddings import create_dataloader

def find_groups(train_data, val_data, aug_indices, feature_extractor, use_classifier_groups=False, num_epochs=5, k=0, max_iter=4, min_group=100, groups=None, **loader_kwargs):
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

    np.savez(f'cmnist_meta_{desc}.npz', predictions=store_pred, embeddings=store_emb, subclass=store_sub, label=store_label, loss=store_loss, data_idx=store_data_idx)
