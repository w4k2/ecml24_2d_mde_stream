import numpy as np
from utils import generate_streams
from strlearn.metrics import balanced_accuracy_score as bac
from mde import STML
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler


random_state = 1410
replications = 5
n_chunks = 1000

streams = generate_streams(random_state, replications)

results = []

def worker(stream):
# for stream_id, stream in enumerate(streams):
    print("Start: %s" % (stream))
    
    """
    Model
    """
    num_classes = 2
    batch_size = 8
    num_epochs = 1
    weights = ResNet18_Weights.IMAGENET1K_V1
    
    model = resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device("mps")
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    weights = torch.from_numpy(np.array([0.05, 0.95])).float().to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    """
    """
    # dis = KBinsDiscretizer(n_bins=3)
    scaler = MinMaxScaler()
    for c in range(n_chunks):
        X, y = stream.get_chunk()
        
        # X = dis.fit_transform(X).toarray().astype(int)
        X = scaler.fit_transform(X)
        X = np.digitize(X, np.linspace(0, 1, 256)).astype(int)
        
        X_stml = STML(X, verbose=False, size=(50, 50), n_cols=5)
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
            score =  bac(y_stml.numpy(), preds)
            results.append(score)
            
            model.train()
            for epoch in range(num_epochs):
                for i, batch in enumerate(data_loader, 0):
                    inputs, labels = batch

                    optimizer.zero_grad()

                    outputs = model(inputs.to(device))
                    loss = criterion(outputs.to(device), labels.to(device))
                    loss.backward()
                    optimizer.step()
            
    np.save("results/stml_synth/%s_discrete" % stream, results)
    print("End: %s" % (stream))
    
    
jobs = []
if __name__ == '__main__':
    for stream in streams:
        p = multiprocessing.Process(target=worker, args=(stream,))
        jobs.append(p)
        p.start()