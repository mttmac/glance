from VAE1D import *
from scipy.stats import multivariate_normal

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
    
    path_params = (date, desc, n_latent, kl_weight)
    path = "../models/{}-{}/best_model-{}-{}.pt".format(*path_params)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    print("Checkpoint Performance:")
    print("Validation loss: {:.3f}".format(checkpoint['val_loss']))
    print("Epoch: {}".format(checkpoint['epoch']))
    return model


def load_clusters():
    params = (date, desc)
    means = np.load('../models/{}-{}/cluster_means.npy'.format(*params))
    covars = np.load('../models/{}-{}/cluster_covars.npy'.format(*params))
    threshold = np.load('../models/{}-{}/threshold.npy'.format(*params))
    return means, covars, threshold


def get_random_samples(n=None):
    dl = load_datasets(Path('../data/hydraulic/{}'.format(desc)))[2]
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
    # Manually calculate the log likelihood of belonging to any cluster
    # Returns max as a lower bound
    prob = np.zeros(means.shape[0])
    for k in range(means.shape[0]):
        prob[k] = multivariate_normal.logpdf(latent,
                                             mean=means[k, :],
                                             cov=covars[k, :, :])
    return prob.max()

    
def detect_anomaly(log_prob, threshold):
    return int(log_prob < threshold)
        

# Update for flask app
# def plot_cycle(X, cycle=0, generated=False, fig=None):
#     if fig is None:
#         fig = plt.figure()
#     else:
#         fig.clf()
#         plt.figure(fig.number)
#     for i in range(X.shape[0]):
#         plt.plot(X[i, :])
#     if not generated:
#         plt.title(f'Input sensor data for cycle {cycle + 1}')
#     else:
#         plt.title(f'Generated sensor data for cycle {cycle + 1}')
#     fig.canvas.draw()
#     return fig
        

# def plot_anomalies(status, cycle, fig=None):
#     if fig is None:
#         fig = plt.figure()
#     else:
#         fig.clf()
#         plt.figure(fig.number)
#     plt.bar(range(1, len(status) + 1), status)
#     limit = 6
#     count = cycle + 1
#     anoms = status.sum()
#     cycle = 'cycle' if count == 1 else 'cycles'
#     anomaly = 'anomaly' if anoms == 1 else 'anomalies'
#     title = f"{count} {cycle}, {int(anoms)} {anomaly}"
#     if count > limit and anoms > limit / 2:
#         title = title + ': MAINTENANCE REQUIRED'
#     plt.title(title)
#     plt.ylim([0, 1])
#     plt.xticks(range(1, count + 1))
#     plt.ylabel('Anomaly')
#     plt.xlabel('Cycle')
#     fig.canvas.draw()
#     return fig
    
    
# def stream(model, means, covars, threshold):
#     threshold = 0
#     X_fig, X_hat_fig, stat_fig = None, None, None
#     n = 20
#     status = np.zeros(n)
#     data, targets = get_random_samples(n)
#     for i in range(n):
#         X = data[i, :, :]
#         latent, X, X_hat = compute_latent(X, model)
#         log_prob = compute_log_prob(latent, means, covars)
#         status[i] = detect_anomaly(log_prob, threshold)
#         X_fig = plot_cycle(X, cycle=i, fig=X_fig)
#         X_hat_fig = plot_cycle(X_hat, cycle=i, generated=True, fig=X_hat_fig)
#         stat_fig = plot_anomalies(status, cycle=i, fig=stat_fig)
#         sleep(2)
    