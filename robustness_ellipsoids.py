from xml.etree import ElementTree
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style as style
from matplotlib.backends.backend_pdf import PdfPages
import seaborn.apionly as sns
import scipy.special as ss
import math
import robustness_calc_cube
import pandas as pd
plt.style.use('ggplot')
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12


def read_input(inp_fil):
    document = ElementTree.parse(inp_fil)
    lims = []
    for item in document.find('parameters').getchildren():
        lims.append([item.find('distribution').text, item.find('start').text, item.find('end').text])
    init = len(document.find('initial').getchildren())
    return lims, init


def read_posterior(post_file, init):
    data = np.genfromtxt(post_file, delimiter=" ")
    post = np.delete(data, 0, 1)
    for j in range(init):
        post = np.delete(post, -1, 1)
    post_5 = np.percentile(post, 1, axis=0)
    post_95 = np.percentile(post, 99, axis=0)
    indeces_to_del = []
    for k in range(post.shape[1]):
        for j in range(post.shape[0]):
            if post[j, k] < post_5[k]:
                indeces_to_del.append(j)
            elif post[j, k] > post_95[k]:
                indeces_to_del.append(j)
    indeces_to_del = sorted(list(set(indeces_to_del)))
    for index in reversed(indeces_to_del):
        post = np.delete(post, index, 0)
    return post


def calc_prior_hypercuboid(lims):
    plevs = []
    for j in lims:
        if j[0] == 'constant':
            continue
        else:
            plev = float(j[2]) - float(j[1])
            plevs.append(plev)
    prior_vol = reduce(lambda x, y: x * y, plevs)
    return prior_vol


def calc_cov(data):
    np.cov(data.T)
    return np.cov(data.T)


def calculate_vol(covar, dimensions):
    #Assume confidence interval = 0.025 (99.7% data)
    Chi_sq_tab = {'3': 9.348, '4': 11.143, '5': 12.833, '6': 14.449, '7': 16.013, '8': 17.535, '9': 19.023, '10': 20.483}
    V = ((2.0 * np.power(math.pi, (dimensions / 2.0))) / (dimensions * ss.gamma(dimensions / 2.0))) * (np.power(Chi_sq_tab[str(dimensions)], (dimensions / 2.0))) * np.sqrt(np.linalg.det(covar))
    return V


def normalize_rob(tmp_robustness):
    robustness = []
    tot_rob = reduce(lambda x, y: x + y, tmp_robustness)
    for r in tmp_robustness:
        robustness.append(r/tot_rob)
    return robustness

models = 2
repeats = 3
robustness_all = np.zeros([models*repeats, 4])
mod = 'sde'
inp_fils = [mod+'/input_file_pop_sde_1.xml', mod+'/input_file_pop_sde_2.xml']

post_files = [[mod +'/run1/results_example4_sde_1/results_immigration-death/Population_3/data_Population3.txt',
                mod + '/run1/results_example4_sde_2/results_logistic/Population_3/data_Population3.txt'],
              [mod+'/run2/results_example4_sde_1/results_immigration-death/Population_3/data_Population3.txt',
               mod + '/run2/results_example4_sde_2/results_logistic/Population_3/data_Population3.txt',],
              [mod +'/run3/results_example4_sde_1/results_immigration-death/Population_3/data_Population3.txt',
                  mod+'/run3/results_example4_sde_2/results_logistic/Population_3/data_Population3.txt']]

index = np.arange(models)

n = np.array(['Model 1', 'Model 2'])
models_n = np.repeat(n, repeats, axis=0)
repeats_n =  ['a', 'b', 'c']*repeats
#mjp
#robustness_all[:, 0] = [0.755572108097, 0.758314339397, 0.701667072874, 0.244427891903, 0.241685660603, 0.298332927126]

#ode
#robustness_all[:, 0] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

#sde
robustness_all[:, 0] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

norm_rob = np.zeros([models, repeats])

for i in range(repeats):
    tmp_robustness = []
    for j in range(models):
        lims, init = read_input(inp_fils[j])
        prior_vol = calc_prior_hypercuboid(lims)
        data = read_posterior(post_files[i][j], init)
        covar = calc_cov(data)
        dimensions = len(lims)
        ellipsoid_vol = calculate_vol(covar, dimensions)
        tmp_robustness.append(ellipsoid_vol/prior_vol)
    norm_rob[:, i] = normalize_rob(tmp_robustness)
robustness_all[:, 1] = norm_rob.flatten()
robustness_all[:, 2] = robustness_calc_cube.main_loop(models, inp_fils, post_files)

ix3 = pd.MultiIndex.from_arrays([models_n, repeats_n], names=['model','repeat'])
df = pd.DataFrame({'ABC-SysBio': robustness_all[:,0], 'Ellipsoid': robustness_all[:,1], 'Cuboid':robustness_all[:,2]}, index = ix3)

gp3 = df.groupby(level=('model'))
means = gp3.mean()
errors = gp3.std()
colors = ['#1d91c0', '#225ea8', '#253494']
flatui = ['#7fcdbb', '#1d91c0', '#225ea8', '#253494']
pp = PdfPages("robustness_comparison_ex4_sde2.pdf")
fig, ax = plt.subplots()
means.plot(yerr=errors, ax=ax, kind='bar', colors=flatui)
pp.savefig()
plt.close()
pp.close()

