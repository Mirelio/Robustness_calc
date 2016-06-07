from xml.etree import ElementTree
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import gaussian_kde
import math
from matplotlib.ticker import FormatStrFormatter
from itertools import product, combinations
import mpl_toolkits.mplot3d.axes3d as axes3d


def generate_data(N, lims, d, cov, distrib):
    if distrib == 'normal':
        posterior = random.multivariate_normal([(lims[0][1]-lims[0][0])/2, (lims[1][1]-lims[0][0])/2, (lims[2][1]-lims[0][0])/2], cov, N).T
    elif distrib == 'uniform':
        posterior = random.uniform(low=(lims[0][0] + d / 2), high=(lims[0][1] - d / 2), size=(3, N))
    return posterior


def sample_priors(lims, numb_to_samp):
    samples_list = []
    for i in lims:
        samples = random.uniform(low=i[0], high=i[1], size=(numb_to_samp))
        samples_list.append(samples)
    return asarray(samples_list).T


def five_ninefive(data, f):
    if f == 0:
        return data
    else:
        post_5 = percentile(data, f, axis=1)
        post_95 = percentile(data, (100-f), axis=1)
        indeces_to_del = []
        for k in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[k, j] < post_5[k]:
                    indeces_to_del.append(j)
                elif data[k, j] > post_95[k]:
                    indeces_to_del.append(j)
        indeces_to_del = sorted(list(set(indeces_to_del)))
        for index in reversed(indeces_to_del):
            data = delete(data, index, 1)
    return data


def do_kde(post):
    kde_res = []
    k = 0
    xxs=[]
    for i in range(len(post)):
        #xmin = float(lims[i][1])
        #xmax = float(lims[i][2])
        xmin = amin(post[i])
        xmax = amax(post[i])
        ngrid = int(1000)
        xx = linspace(xmin, xmax, ngrid)
        kernel = gaussian_kde(post[i])
        Z = reshape(kernel(xx).T, xx.shape)
        k += 1
        kde_res.append(Z)
        xxs.append(xx)
    return kde_res, xxs


def accept_reject_allD(samples_list, data, numb_to_samp):
    samp_counter = -1
    accepted = []
    ranges = []
    for k in range(data.shape[0]):
        xmin = amin(data[k])
        xmax = amax(data[k])
        rang = [xmin, xmax]
        ranges.append(rang)

    for sample in samples_list:
        samp_counter += 1
        s = -1
        flag = []
        for j in range(len(sample)):
            s += 1
            if sample[j] > ranges[s][0] and sample[j] < ranges[s][1]:
                flag.append(1)
            else:
                flag.append(0)
        if 0 not in flag:
            accepted.append(samp_counter)
    acc_rate = float(len(accepted))/numb_to_samp
    return acc_rate


def calc_true_robustness(lims, d, cov, variance, distrib, post):
    prior_vol = float(power((lims[0][1]) - (lims[0][0]), 3))
    if distrib == 'normal':
        true_vol_sq = power((2 * math.pi), 1.5) * sqrt(linalg.det(cov))
        radius = sqrt(variance)*3
        #radius = (max(post[0])-min(post[0]))/2
        true_vol_sph = (4/3)*math.pi*(power(radius, 3))
        true_robustness_sq = true_vol_sq / prior_vol
        true_robustness_sph = true_vol_sph / prior_vol
        return true_robustness_sq, true_robustness_sph

    elif distrib == 'uniform':
        true_vol = float(power((lims[0][1] - d/2) - (lims[0][0] + d/2), 3))
        true_robustness = true_vol/prior_vol
        return true_robustness, 0


d = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
cf = [0, 1, 2, 3, 4, 5]
repeats = 10
models = 6
numb_to_samp = 1000
robust_measures = zeros([repeats, len(cf)])
sds = zeros([models, len(cf)])
means = zeros([models, len(cf)])
mean_d_sq = zeros([models, len(cf)])
sd_d_sq = zeros([models, len(cf)])
mean_d_sph = zeros([models, len(cf)])
sd_d_sph = zeros([models, len(cf)])

true_robustness = []
lims = [[[0, 100], [0, 100], [0, 100]], [[0, 100], [0, 100], [0, 100]], [[0, 100], [0, 100], [0, 100]], [[0, 100], [0, 100], [0, 100]], [[0, 100], [0, 100], [0, 100]], [[0, 100], [0, 100], [0, 100]]]
index = arange(models)
fract = []
for de in d:
    fract.append((100.0-de)/100.0)
var = [256, 200, 150, 100, 50, 20]
covar = [[[256, 0, 0], [0, 256, 0], [0, 0, 256 ]], [[200, 0, 0], [0, 200, 0], [0, 0, 200]], [[150, 0, 0], [0, 150, 0], [0, 0, 150]], [[100, 0, 0], [0, 100, 0], [0, 0, 100]], [[50, 0, 0], [0, 50, 0], [0, 0, 50]], [[20, 0, 0], [0, 20, 0], [0, 0, 20]]]
posts = []
distrib = 'uniform' # 'normal' or 'uniform'

for j in range(models):
    means_c = []
    sds_c = []
    m_d_sq = []
    s_d_sq = []
    m_d_sph = []
    s_d_sph = []

    post = generate_data(1000, lims[j], d[j], covar[j], distrib)
    t_r_sq, t_r_sph = calc_true_robustness(lims[j], d[j], covar[j], var[j], distrib, post)
    true_robustness.append(t_r_sq)
    posts.append(post)
    for f in range(len(cf)):
        data = five_ninefive(post, cf[f])
        for i in range(repeats):
            samples_list = sample_priors(lims[j], numb_to_samp)
            robust_measures[i][f] = accept_reject_allD(samples_list, data, numb_to_samp)
        means_c.append(mean(robust_measures[:, f]))
        sds_c.append(std(robust_measures[:, f]))
        di_sq = [x / t_r_sq for x in robust_measures[:, f]]
        di_sph = [x / t_r_sph for x in robust_measures[:, f]]
        m_d_sq.append(mean(di_sq))
        s_d_sq.append(std(di_sq))
        m_d_sph.append(mean(di_sph))
        s_d_sph.append(std(di_sph))
    means[j] = means_c
    sds[j] = sds_c
    mean_d_sq[j] = m_d_sq
    sd_d_sq[j] = s_d_sq
    mean_d_sph[j] = m_d_sph
    sd_d_sph[j] = s_d_sph
print mean_d_sph.T

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + random.randn(len(arr)) * stdev

#'#ffffd9','#edf8b1',,,
#'#ffffcc', '#ffeda0','#fed976',,
colors = ['#c7e9b4','#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494']#,'#081d58']
colors2 = ['#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
mean_dsqT = mean_d_sq.T
mean_dsphT = mean_d_sph.T
#sns.set(style="white", palette="muted", color_codes=True)
sns.set_style("ticks")
pp = PdfPages("robustness_uniform_calc_true.pdf")
fig, ax = plt.subplots(1,1)
#var2 = [x-2 for x in var]
for j in range(len(mean_dsqT)):
    ax.errorbar(rand_jitter(fract), mean_dsqT[j], yerr=sd_d_sq.T[j], fmt='o', color=colors[j],alpha=0.8,ecolor=colors[j], label='%s Cut-off' % (cf[j]/100.0))
#for j in range(len(mean_dsphT)):
#    ax.errorbar(rand_jitter(var2), mean_dsphT[j], yerr=sd_d_sph.T[j], fmt='o', color=colors2[j],alpha=0.8, ecolor=colors2[j], label=' ')
#true_rob = ax.scatter(fract, diff, marker='o', color='#32995F')
l = ax.axhline(y=1,color='black',ls='dashed')
ax.set_xlabel('Fraction')
#ax.set_yscale('log')
#plt.tick_params(axis='y', which='minor')
#ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
#ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.set_xticks(fract)
fract_str = [str(i) for i in fract]
ax.set_xticklabels(fract_str)
#ax.set_xlabel('Fraction')
ax.set_ylabel('Rcalc/Rtrue')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.xlim((17,0))
plt.xlim((1.1, 0.4))

#plt.xlim((1, 0.4))
plt.ylim((0, 2))
plt.tight_layout()
sns.despine()
pp.savefig(bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()
pp.close()
