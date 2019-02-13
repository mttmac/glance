from VAE1D import *
from scipy.stats import multivariate_normal
from time import sleep

import matplotlib.pyplot as plt
plt.style.use('ggplot')

size = 512
n_channels = 14
n_latent = 50
kl_weight = 1

date = '190130'
desc = 'accumulator'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint():
    model = VAE1D(size, n_channels, n_latent)
    model = model.to(device)
    
    path = f'../models/{date}-{desc}/best_model-{n_latent}-{kl_weight}.pt'
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    print("Checkpoint Performance:")
    print(f"Validation loss: {checkpoint['val_loss']:.3f}")
    print(f"Epoch: {checkpoint['epoch']}")
    return model


def load_clusters():
    means = np.load(f'../models/{date}-{desc}/cluster_means.npy')
    covars = np.load(f'../models/{date}-{desc}/cluster_covars.npy')
    threshold = np.load(f'../models/{date}-{desc}/threshold.npy')
    return means, covars, threshold


def get_random_samples(n=None):
    dl = load_datasets(Path(f'../data/hydraulic/{desc}'))[2]
    if n is None:
        n = len(dl)
    for i, (X, y) in enumerate(dl):
        if i == 0:
            data = X
            targets = y
        elif i < n:
            data = torch.cat([data, X], 0)
            targets = torch.cat([targets, y], 0)
        else:
            break
    return data, targets


def compute_latent(X, model):    
    model.eval()
    with torch.no_grad():
        X = X.unsqueeze(0).to(device)
        X_hat, mu, logvar = model(X)
        mu = mu.cpu().detach().numpy()
        X = X[0, :, :].cpu().detach().numpy()
        X_hat = X_hat[0, :, :].cpu().detach().numpy()
    return mu, X, X_hat


def compute_log_prob(latent, means, covars):
    # Manually calculate the probability of belonging to a cluster
    prob = np.zeros(1)
    for k in range(means.shape[0]):
        prob += multivariate_normal.pdf(latent,
                                        mean=means[k, :],
                                        cov=covars[k, :, :])
    if prob == 0:
        prob = np.array(1e-100)
    return np.log(prob)

    
def detect_anomaly(log_prob, threshold):
    return int(log_prob < threshold)
        

def plot_cycle(X, cycle=0, generated=False, fig=None):
    if fig is None:
        fig = plt.figure()
    else:
        fig.clf()
        plt.figure(fig.number)
    for i in range(X.shape[0]):
        plt.plot(X[i, :])
    if not generated:
        plt.title(f'Input sensor data for cycle {cycle + 1}')
    else:
        plt.title(f'Generated sensor data for cycle {cycle + 1}')
    fig.canvas.draw()
    return fig
        

def plot_anomalies(status, cycle, fig=None):
    if fig is None:
        fig = plt.figure()
    else:
        fig.clf()
        plt.figure(fig.number)
    plt.bar(range(1, len(status) + 1), status)
    limit = 6
    count = cycle + 1
    anoms = status.sum()
    cycle = 'cycle' if count == 1 else 'cycles'
    anomaly = 'anomaly' if anoms == 1 else 'anomalies'
    title = f"{count} {cycle}, {int(anoms)} {anomaly}"
    if count > limit and anoms > limit / 2:
        title = title + ': MAINTENANCE REQUIRED'
    plt.title(title)
    plt.ylim([0, 1])
    plt.xticks(range(1, count + 1))
    plt.ylabel('Anomaly')
    plt.xlabel('Cycle')
    fig.canvas.draw()
    return fig
    
    
def stream(model, means, covars, threshold):
    threshold = 0
    X_fig, X_hat_fig, stat_fig = None, None, None
    n = 20
    status = np.zeros(n)
    data, targets = get_random_samples(n)
    for i in range(n):
        X = data[i, :, :]
        latent, X, X_hat = compute_latent(X, model)
        log_prob = compute_log_prob(latent, means, covars)
        status[i] = detect_anomaly(log_prob, threshold)
        X_fig = plot_cycle(X, cycle=i, fig=X_fig)
        X_hat_fig = plot_cycle(X_hat, cycle=i, generated=True, fig=X_hat_fig)
        stat_fig = plot_anomalies(status, cycle=i, fig=stat_fig)
        sleep(2)
    