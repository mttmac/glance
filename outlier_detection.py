criterion = VAE2DLoss(kl_weight=0.01)
test_loader = test_dl
image_size = 128
val_loader = val_dl

############################# ANOMALY SCORE DEF ##########################
def get_vae_score(vae, image, L=5):
    """
    The vae score for a single image, which is basically the loss
    :param image: [1, 3, 256, 256]
    :return (vae loss, KL, reconst_err)
    """
    image_batch = image.expand(L,
                               image.size(1),
                               image.size(2),
                               image.size(3))
    reconst_batch, mu, logvar = vae.forward(image_batch)
    vae_loss, loss_details = criterion(reconst_batch, image_batch, mu, logvar)
    return vae_loss, loss_details['KL'], -loss_details['logp']

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

############################# END OF ANOMALY SCORE ###########################

# Define the number of samples of each score
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


# MAIN LOOP
score_names = ['reconst_score', 'KL_score', 'vae_score',
               'iwae_reconst_score', 'iwae_KL_score', 'iwae_score']
classes = test_loader.dataset.classes
scores = {(score_name, cls): [] for (score_name, cls) in product(score_names,
                                                                 classes)}
model.eval()
with torch.no_grad():
    for idx, (image, target) in tqdm(enumerate(test_loader)):
        cls = classes[target.item()]
        image = image.to(device)

        score = compute_all_scores(vae=model, image=image)
        for name in score_names:
            scores[(name, cls)].append(score[name])

# display the mean of scores
means = np.zeros([len(score_names), len(classes)])
for (name, cls) in product(score_names, classes):
    means[score_names.index(name), classes.index(cls)] = sum(scores[(name, cls)]) / len(scores[(name, cls)])
df_mean = pd.DataFrame(means, index=score_names, columns=classes)
print("###################### MEANS #####################")
print(df_mean)


classes.remove('NV')
auc_result = np.zeros([len(score_names), len(classes)])
# get auc roc for each class
for (name, cls) in product(score_names, classes):
    normal_scores = scores[(name, 'NV')]
    abnormal_scores = scores[(name, cls)]
    y_true = [0]*len(normal_scores) + [1]*len(abnormal_scores)
    y_score = normal_scores + abnormal_scores
    auc_result[score_names.index(name), classes.index(cls)] = roc_auc_score(y_true, y_score)
"""
# add auc roc against all diseases
for name in score_names:
    normal_scores = scores[(name, 'NV')]
    abnormal_scores = np.concatenate([scores[(name, cls)]for cls in classes]).tolist()
    y_true = [0]*len(normal_scores) + [1]*len(abnormal_scores)
    y_score = normal_scores + abnormal_scores
    auc_result[score_names.index(name), -1] = roc_auc_score(y_true, y_score)
"""

df = pd.DataFrame(auc_result, index=score_names, columns=classes)
# display
print("###################### AUC ROC #####################")
print(df)
print("####################################################")
# df.to_csv(args.out_csv)

# fit a gamma distribution
# _, val_loader = load_vae_train_datasets(image_size, args.data, 32)
model.eval()
all_reconst_err = []
num_val = len(val_loader.dataset)
with torch.no_grad():
    for img, _ in tqdm(val_loader):
        img = img.to(device)

        # compute output
        recon_batch, mu, logvar = model(img)
        loss, loss_details = criterion.forward(recon_batch, img, mu, logvar, reduce=False)
        reconst_err = -loss_details['logp']
        all_reconst_err += reconst_err.tolist()

fit_alpha, fit_loc, fit_beta=stats.gamma.fit(all_reconst_err)

# using gamma for outlier detection
# get auc roc for each class
LARGE_NUMBER = 1e30

def get_gamma_score(scores):
    result = -stats.gamma.logpdf(scores, fit_alpha, fit_loc, fit_beta)
    # replace inf in result with largest number
    result[result == np.inf] = LARGE_NUMBER
    return result

auc_gamma_result = np.zeros([1, len(classes)])
name = 'reconst_score'
for cls in classes:
    normal_scores = get_gamma_score(scores[(name, 'NV')]).tolist()
    abnormal_scores = get_gamma_score(scores[(name, cls)]).tolist()
    y_true = [0]*len(normal_scores) + [1]*len(abnormal_scores)
    y_score = normal_scores + abnormal_scores
    auc_gamma_result[0, classes.index(cls)] = roc_auc_score(y_true, y_score)

# for all class
"""
normal_scores = get_gamma_score(scores[(name, 'NV')]).tolist()
abnormal_scores = np.concatenate([get_gamma_score(scores[(name, cls)]) for cls in classes]).tolist()
y_true = [0]*len(normal_scores) + [1]*len(abnormal_scores)
y_score = normal_scores + abnormal_scores
auc_gamma_result[0, -1] = roc_auc_score(y_true, y_score)
"""

df = pd.DataFrame(auc_gamma_result, index=['gamma score'], columns=classes)

# display
print("###################### AUC ROC GAMMA #####################")
print(df)
print("##########################################################")
