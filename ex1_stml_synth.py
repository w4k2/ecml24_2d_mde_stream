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


random_state = 1410
replications = 5
n_chunks = 3000

streams = generate_imb_streams(random_state, replications)

metrics=[recall, precision, precision_score, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2, bac]

def worker(stream):
# for stream_id, stream in enumerate(streams):
    print("Start: %s" % (stream))
    results = []
    
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
    
    if "d95" in str(stream):
        imb_weights = torch.from_numpy(np.array([0.05, 0.95])).float().to(device)
    if "d85" in str(stream):
        imb_weights = torch.from_numpy(np.array([0.15, 0.85])).float().to(device)
        
    criterion = nn.CrossEntropyLoss(weight=imb_weights)
    """
    """
    
    for c in range(n_chunks):
        X, y = stream.get_chunk()
        
        X_stml = STML(X, verbose=False, size=(50, 50))
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
            # score =  bac(y_stml.numpy(), preds)
            scores = [metric(y_stml.numpy(), preds) for metric in metrics]
            results.append(scores)
            
            model.train()
            for epoch in range(num_epochs):
                for i, batch in enumerate(data_loader, 0):
                    inputs, labels = batch

                    optimizer.zero_grad()

                    outputs = model(inputs.to(device))
                    loss = criterion(outputs.to(device), labels.to(device))
                    loss.backward()
                    optimizer.step()
    
    results = np.array(results)
    np.save("results/stml_synth/%s_imb_notransfer" % stream, results)
    print("End: %s" % (stream))
    
    
jobs = []
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=15)
    for stream in streams:
        pool.apply_async(worker, args=(stream,))
    pool.close()
    pool.join()