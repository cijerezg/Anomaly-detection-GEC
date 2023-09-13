import numpy as np
import scipy.special as special
from itertools import combinations_with_replacement as co


SAMPLES = 100

# Toy scores. Put your real scores here.
score1 = np.random.rand(SAMPLES)
score2 = np.random.rand(SAMPLES)


per = score1.shape[0] // 100 # Calculate percentage for borders.
# This uses three borders, but they may be modified and more may be added.
BORDERS = [(0, per*90), (per*90, per*98), (per*98, per*100)]


ord_sco1 = np.argsort(score1)
index = np.arange(score1.size)

score1 = np.concatenate(
    (score1[ord_sco1, np.newaxis], index[:, np.newaxis]), 1)
score2 = np.concatenate(
    (score2[ord_sco1, np.newaxis], index[:, np.newaxis]), 1)

scores = [score1, score2]


def score_preprocessing(scores):
    ord_scores = []
    for sc in scores:
        ord_scores.append(sc[np.argsort(sc[:, 0]), :])
    classes = [[] for i in range(2)]
    for j in range(3):
        for i, ord_s in enumerate(ord_scores):
            classes[i].append(ord_s[BORDERS[j][0]:BORDERS[j][1]])
    means = [[] for i in range(3)]
    stds = [[] for i in range(3)]
    for j in range(3):
        c_i = set(classes[0][j][:, 1]).intersection(set(classes[1][j][:, 1]))
        for k in range(2):
            mean = np.mean(classes[k][j][:, 0])
            means[j].append(mean)
            std = np.std(classes[k][j][:, 0])
            stds[j].append(std)
            arr = classes[k][j]
            inds = np.asarray(list(set(arr[:, 1]) - c_i), dtype=int)
            classes[k][j] = arr[np.isin(arr[:, 1], inds)]
    return means, stds, classes


def reindexing(classes):
    state = 0
    keys = []
    vals = []
    for i in range(3):
        key = classes[0][i][:, 1]
        val = np.arange(state, state + key.size)
        state = state + key.size
        keys.extend(key)
        vals.extend(val)
        classes[0][i][:, 1] = val
    my_dict = dict(zip(keys, vals))
    for j in range(3):
        new_keys = classes[1][j][:, 1]
        new_vals = list(map(my_dict.get, new_keys))
        classes[1][j][:, 1] = new_vals
    return len(vals), classes


def gaussian_ev(points, mu, std):
    arg = (points-mu)/(std*np.sqrt(2))
    fx = 1/2*(1+special.erf(arg))
    return fx


def integral_weights(ps, mus, stds):
    weights = 0
    for mu, std in zip(mus, stds):
        dists = []
        for point in ps:
            dists.append(gaussian_ev(point, mu, std))
        weight = np.abs(np.subtract.outer(dists[0], dists[1]))
        weights += weight
    return weights


def ind_sign(ps, same):
    ratio = np.divide.outer(ps[0]+1, ps[1]+1)
    if same == False:
        rat = np.where(ratio > 1, -2, 2)
    else:
        rat = np.where(ratio > 1, -1, 1)
    return rat


def tau(means, stds, classes, length):
    coeff = np.zeros((length, length, 2))
    sets = [0, 1, 2]
    for k in range(2):
        clas = classes[k]
        for i, j in co(sets, 2):
            offset = 0
            if i - j > 1:
                offset = 1
            ps = [clas[i][:, 0], clas[j][:, 0]]
            mus = [means[i][k], means[j][k]]
            sta = [stds[i][k], stds[j][k]]
            ws = integral_weights(ps, mus, sta) + offset
            in_t = [clas[i][:, 1], clas[j][:, 1]]
            same = False
            if i == j:
                same = True
            ratios = ind_sign(ps, same)
            xc = np.repeat(in_t[0][:, np.newaxis], in_t[1].size, axis=1)
            yc = np.repeat(in_t[1][np.newaxis, :], in_t[0].size, axis=0)
            xc = xc.astype(int).flatten()
            yc = yc.astype(int).flatten()
            part_coef = ws * ratios
            coeff[xc, yc, k] = part_coef.flatten()
    return coeff


def coef_prep(coef):
    for i in range(2):
        coef[:, :, i] = np.where(
            coef[:, :, i] != 0, coef[:, :, i], -np.transpose(coef[:, :, i]))
        coef[:, :, i] = np.triu(coef[:, :, i], 1)
    return coef


def score_computing(mus, sigmas, classes, length):
    coef = tau(mus, sigmas, classes, length)
    coef = coef_prep(coef)
    tau_s = np.sum(coef[:, :, 0] * coef[:, :, 1])
    return tau_s, coef


mus, sigmas, classes = score_preprocessing(scores)

length, classes = reindexing(classes)

mu_norm1 = [[mus[j][0] for i in range(2)] for j in range(3)]
mu_norm2 = [[mus[j][1] for i in range(2)] for j in range(3)]
si_norm1 = [[sigmas[j][0] for i in range(2)] for j in range(3)]
si_norm2 = [[sigmas[j][1] for i in range(2)] for j in range(3)]

class_norm1 = [[classes[0][i] for i in range(3)] for j in range(2)]
class_norm2 = [[classes[1][i] for i in range(3)] for j in range(2)]


tau_s1, coef_s1 = score_computing(mu_norm1, si_norm1, class_norm1, length)
tau_co, coef_co = score_computing(mus, sigmas, classes, length)
tau_s2, coef_s2 = score_computing(mu_norm2, si_norm2, class_norm2, length)

tau_tot = tau_co / (np.sqrt(tau_s1 * tau_s2))

