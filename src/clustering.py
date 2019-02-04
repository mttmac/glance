from VAE1D import *

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.style.use('ggplot')

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

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


# from pdb import set_trace
def draw_ellipse(position, covariance, pca, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = plt.gca()
    
    # Convert covariance to principal axes
    position = pca.transform(position[None, :]).ravel()
    covariance = pca.transform(covariance)
    covariance = pca.transform(covariance.T)
    U, s, Vt = np.linalg.svd(covariance)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2 * np.sqrt(s)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position,
                             nsig * width,
                             nsig * height,
                             angle,
                             **kwargs))
        
def plot_clusters(latents, gmm, show_pdf=True, ax=None):
    # Project to 2D for plotting
    if ax is None:
        plt.figure()
        ax = plt.gca()
    
    pca = PCA(n_components=2)
    lat_pca = pca.fit_transform(latents)
    clusters = gmm.predict(latents)
    
    ax.scatter(lat_pca[:, 0], lat_pca[:, 1],
               c=clusters, cmap='viridis',
               zorder=2)
    ax.axis('equal')
    
    if show_pdf:
        w_max = gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            draw_ellipse(pos, covar, pca, alpha=0.2 * w / w_max)

            
def calculate_log_probs(latents, gmm, method='bounded'):
    # Manually calculate the probability of each point belonging to a cluster
    means = gmm.means_
    covars = gmm.covariances_
    weights = gmm.weights_
    
    if method == 'bounded':
        # Higher numerical precision but only returns the lower bound on probability
        log_probs = np.zeros([latents.shape[0], means.shape[0]])
        for k in range(means.shape[0]):
            # Note: in log space addition is multiplication
            log_probs[:, k] = multivariate_normal.logpdf(latents,
                                                         mean=means[k, :],
                                                         cov=covars[k, :, :])
        log_probs = log_probs.max(axis=1)
    elif method == 'clipped':
        # Clipped on low end due to numerical round off but uses sum of all cluster probabilities
        probs = np.zeros(latents.shape[0])
        for k in range(means.shape[0]):
            w = weights[k] / weights.max()
            probs += w * multivariate_normal.pdf(latents,
                                                 mean=means[k, :],
                                                 cov=covars[k, :, :])
        # Set all zero probabilities to min non-zero probability
        probs[probs == 0] = probs[probs != 0].min()
        log_probs = np.log(probs)
    else:
        print("Method not recognized, must be 'bounded' or 'clipped'.")
    
    return log_probs
