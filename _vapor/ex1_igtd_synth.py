import numpy as np
from utils import generate_streams
from strlearn.metrics import balanced_accuracy_score as bac
from mde import STML, min_max_transform, table_to_image_array
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
    
    for c in range(n_chunks):
        X, y = stream.get_chunk()
        
        # IGTD
        max_step = 10000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
        val_step = 100  # The number of iterations for determining algorithm convergence. If the error reduction rate
                        # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.
        fea_dist_method = 'Euclidean'
        image_dist_method = 'Euclidean'
        error = 'abs'

        n_pixels = int(np.ceil(np.sqrt(X.shape[1])))

        norm_data = min_max_transform(X)
        X_igtd = table_to_image_array(norm_data, [n_pixels, n_pixels], fea_dist_method, image_dist_method, max_step, val_step, error)
        X_igtd = np.moveaxis(X_igtd, 2, 0)
        X_igtd = np.repeat(X_igtd[:, :, :, np.newaxis], 3, axis=3).astype(np.uint8)
        X_igtd = np.moveaxis(X_igtd, 3, 1)
        
        X_stml = torch.from_numpy(X_igtd).float()
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
            
    np.save("results/igtd_synth/%s" % stream, results)
    print("End: %s" % (stream))
    
    
jobs = []
if __name__ == '__main__':
    for stream in streams:
        p = multiprocessing.Process(target=worker, args=(stream,))
        jobs.append(p)
        p.start()