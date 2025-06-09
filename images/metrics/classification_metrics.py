import numpy as np

import torch
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


def sample_features(model, data_loader, device='cuda', deterministic=False, num_random_samples=1):
    
    features = []
    labels = []

    with torch.no_grad():

        model.eval()

        pbar = tqdm(iterable=data_loader, disable=True)

        for iter_index, (batch, batch_labels) in enumerate(pbar):

            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)
            real_batch = batch.to(device)

            b_size = real_batch.shape[0]
            real_mu, real_logvar = model.encode(real_batch)


            for _ in range(num_random_samples):

                std = torch.exp(0.5 * real_logvar)
                eps = torch.randn_like(std).to(device)
                if deterministic:
                    z = real_mu
                else:
                    z = real_mu + eps*std
                features.append(z.data.cpu().numpy())
                labels.append(batch_labels)

    labels = np.concatenate(labels)
    features = np.concatenate(features)
    
    model.train()

    return labels, features


def fit_classifier(clf, train_features, train_labels, test_features, test_labels):

    # fit on train
    clf.fit(train_features, train_labels)
    # test on test
    test_acc = clf.score(test_features, test_labels)

    # y_pred = clf.predict(test_features)
    # cm = confusion_matrix(test_labels, y_pred)
    # print(cm)

    return test_acc


def classification_performance(model, train_data_loader, test_data_loader, device='cuda', 
                                 num_random_train_samples=1,  
                                 num_random_test_samples=1, 
                                 normalize=True):
    
    classifiers = {'svm_few': SVC(kernel="linear", C=0.025, max_iter=2_000), # default in LinearSVC is 1000
                   'svm_many': SVC(kernel="linear", C=0.025, max_iter=10_000),

                   'knn_few': KNeighborsClassifier(n_neighbors=5), 
                   'knn_many': KNeighborsClassifier(n_neighbors=100)}
    
    train_labels, train_features = sample_features(model, train_data_loader, deterministic=False,
                                                          device=device, num_random_samples=num_random_train_samples)
    test_labels, test_features = sample_features(model, test_data_loader,  deterministic=False, 
                                                        device=device, num_random_samples=num_random_test_samples)


    if normalize:
        # https://forecastegy.com/posts/does-svm-need-feature-scaling-or-normalization/
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

    test_acc = {}
    for (clf_name, clf) in classifiers.items():

        test_acc[clf_name] = fit_classifier(clf, train_features, train_labels, 
                                                 test_features, test_labels)
    
        print(clf_name, test_acc[clf_name], '\n')

    return test_acc