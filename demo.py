from VAE1D import *
import matplotlib.pyplot as plt

size = 512
n_channels = 14
n_latent = 50

date = '190129'

def load_checkpoint(model, device):
    checkpoint = torch.load(f'models/{date}-hydraulic/best_model-{n_latent}.pt', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    print("Checkpoint Performance:")
    print(f"Validation loss: {checkpoint['val_loss']:.3f}")
    print(f"Epoch: {checkpoint['epoch']}")
    
    return model


def get_random_samples(dl, n=20):
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


def list_target_classes(dl):
    classes = dl.dataset.classes
    for i, clss in enumerate(classes):
        print(f"{i} = {clss}")
    
    
def show_plot(data):
    assert data.shape == (14, 512), "Size error in plotting"
    for i in range(data.shape[0]):
        plt.plot(data[i, :])
    # TODO: make into a grid of plots


# Unchanged
def _log_mean_exp(x, dim):
    """
    A numerical stable version of log(mean(exp(x)))
    :param x: The input
    :param dim: The dimension along which to take mean with
    """
    # m [dim1, 1]
    m, _ = torch.max(x, dim=dim, keepdim=True)

    # x0 [dm1, dim2]
    x0 = x - m

    # m [dim1]
    m = m.squeeze(dim)

    return m + torch.log(torch.mean(torch.exp(x0),
                                    dim=dim))

def get_iwae_score(vae, image, L=5):
    """
    The vae score for a single image, which is basically the loss
    :param image: [1, 3, 256, 256]
    :return scocre: (iwae score, iwae KL, iwae reconst).
    """
    # [L, 3, 256, 256]
    image_batch = image.expand(L,
                               image.size(1),
                               image.size(2),
                               image.size(3))

    # [L, z_dim, 1, 1]
    mu, logvar = vae.encode(image_batch)
    eps = torch.randn_like(mu)
    z = mu + eps * torch.exp(0.5 * logvar)
    kl_weight = criterion.kl_weight
    # [L, 3, 256, 256]
    reconst = vae.decode(z)
    # [L]
    log_p_x_z = -torch.sum((reconst - image_batch).pow(2).reshape(L, -1),
                          dim=1)

    # [L]
    log_p_z = -torch.sum(z.pow(2).reshape(L, -1), dim=1)

    # [L]
    log_q_z = -torch.sum(eps.pow(2).reshape(L, -1), dim=1)

    iwae_score = -_log_mean_exp(log_p_x_z + (log_p_z - log_q_z)*kl_weight, dim=0)
    iwae_KL_score = -_log_mean_exp(log_p_z - log_q_z, dim=0)
    iwae_reconst_score = -_log_mean_exp(log_p_x_z, dim=0)

    return iwae_score, iwae_KL_score, iwae_reconst_score


def compute_all_scores(vae, image):
    """
    Given an image compute all anomaly score
    return (reconst_score, vae_score, iwae_score)
    """
    vae_loss, KL, reconst_err = get_vae_score(vae, image=image, L=15)
    iwae_loss, iwae_KL, iwae_reconst = get_iwae_score(vae, image, L=15)
    result = {'reconst_score': reconst_err.item(),
              'KL_score': KL.item(),
              'vae_score': vae_loss.item(),
              'iwae_score': iwae_loss.item(),
              'iwae_KL_score': iwae_KL.item(),
              'iwae_reconst_score': iwae_reconst.item()}
    return result
