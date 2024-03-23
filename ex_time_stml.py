import numpy as np
from utils import generate_imb_streams
from strlearn.metrics import balanced_accuracy_score as bac, recall, precision, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2
from sklearn.metrics import precision_score
from mde import STML
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
import strlearn as sl
from time import time


n_chunks=110
chunk_size=250
n_features = 64
n_cols = 5
# 8 - 50 None, 16 - 80 3, 32 - 110 3, 64 - 150 5
stml_size = 150

stream = sl.streams.StreamGenerator(n_chunks=n_chunks, chunk_size=chunk_size, n_features=n_features, n_informative=n_features, n_redundant=0, random_state=1410, n_drifts=1, weights=[.9, .1])


# for stream_id, stream in enumerate(streams):
print("Start: %s" % (stream))
"""
Model
"""
num_classes = 2
batch_size = 8
num_epochs = 1
# weights = ResNet18_Weights.IMAGENET1K_V1
weights = None

model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("mps")
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


imb_weights = torch.from_numpy(np.array([0.1, 0.9])).float().to(device)
    
criterion = nn.CrossEntropyLoss(weight=imb_weights)
"""
"""
time_results = np.zeros((1, 110))
for c in range(n_chunks):
    X, y = stream.get_chunk()
    
    start = time()
    
    X_stml = STML(X, verbose=False, size=(stml_size, stml_size), n_cols=n_cols)
    
    # plt.imshow(X_stml[0])
    # plt.savefig("wuj.png")
    # exit()
    
    X_stml = np.swapaxes(X_stml, 1, 3)
        # Second swap for right dimensions
    X_stml = np.swapaxes(X_stml, 2, 3)
    X_stml = torch.from_numpy(X_stml).float()
    y_stml = torch.from_numpy(y).long()
    
    stml_dataset = TensorDataset(X_stml, y_stml)
    data_loader = DataLoader(stml_dataset, batch_size=batch_size, shuffle=True)
    
    if c==0:
        model.train()
        for epoch in range(num_epochs):
            for i, batch in enumerate(data_loader, 0):
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
                
    else:
        model.eval()
        logits = model(X_stml.to(device))
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
        preds = np.argmax(probs, 1)
        
        model.train()
        for epoch in range(num_epochs):
            for i, batch in enumerate(data_loader, 0):
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
    
    elapsed = time() - start
    time_results[0, c] = elapsed

np.save("results/time/time_stml_64", time_results)
print("End: %s" % (stream))