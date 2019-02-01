from VAE1D import *

from sklearn.metrics import roc_auc_score, f1_score
from itertools import product
import scipy.stats as stats

size = 512
n_channels = 14
n_latent = 50
kl_weight = 1

date = '190130'
desc = 'accumulator'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_checkpoint(model, device):
    path = f'../models/{date}-{desc}/best_model-{n_latent}-{kl_weight}.pt'
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    print("Checkpoint Performance:")
    print(f"Validation loss: {checkpoint['val_loss']:.3f}")
    print(f"Epoch: {checkpoint['epoch']}")
    return model


def compute_scores(X, model, criterion):
    assert X.shape[0] == 1, "Must compute score for one sample at a time"
    X_hat, mu, logvar = model(X)
    loss, loss_desc = criterion(X_hat, X, mu, logvar, reduce=False)
    score = {'loss': loss.item(),
             'KL': loss_desc['KL'].item(),
             'error': -loss_desc['logp'].item()}
    return score


def score(dl, model, criterion):
    score_names = ['loss', 'KL', 'error']
    classes = dl.dataset.classes
    scores = {(name, cls): [] for name in score_names for cls in classes}
    
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(dl)):
            X = X.to(device)
            for j in range(X.shape[0]):
                data = X[j, :].unsqueeze(0)
                cls = classes[y[j].item()]
                score = compute_scores(data, model, criterion)
                for name in score_names:
                    scores[(name, cls)].append(score[name])
    return scores


def auc_score(dl, scores):
    score_name = 'error'
    classes = dl.dataset.classes
    y_true = []
    y_score = []
    for i, cls in enumerate(classes):
        cls_score = scores[(score_name, cls)]
        y_true.extend([i] * len(cls_score))
        y_score.extend(cls_score)
    y_true = (np.array(y_true) == 0).astype(np.bool)  # True when fault
    return roc_auc_score(y_true, y_score)


def compute_latents(dl, model):
    latents = []
    targets = []
    
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(dl)):
            X = X.to(device)
            for j in range(X.shape[0]):
                data = X[j, :].unsqueeze(0)
                X_hat, mu, logvar = model(data)
                
                latents.append(mu.cpu().detach().numpy())
                targets.append(y[j].item())
    
    latents = np.array(latents).reshape(len(latents), -1)
    targets = np.array(targets)
    return latents, targets


def compute_latent_and_loss(dl, model, criterion):
    latents = []
    kl = []
    error = []
    targets = []
    
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(dl)):
            X = X.to(device)
            for j in range(X.shape[0]):
                data = X[j, :].unsqueeze(0)
                X_hat, mu, logvar = model(data)
                loss, loss_desc = criterion(X_hat, X,
                                            mu, logvar,
                                            reduce=False)
                
                latents.append(mu.cpu().detach().numpy())
                kl.append(loss_desc['KL'].item())
                error.append(-loss_desc['logp'].item())
                targets.append(y[j].item())
    
    latents = np.array(latents).reshape(len(latents), -1)
    kl = np.array(kl)
    error = np.array(error)
    targets = np.array(targets)
    return latents, kl, error, targets