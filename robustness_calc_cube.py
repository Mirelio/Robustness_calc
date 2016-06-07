from xml.etree import ElementTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


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


def define_post_hypercuboid(data):
    ranges = []
    plevres = []
    for k in range(data.shape[1]):
        xmin = np.amin(data[:, k])
        xmax = np.amax(data[:, k])
        rang = [xmin, xmax]
        plevra = xmax - xmin
        ranges.append(rang)
        plevres.append(plevra)
    cuboid_vol = reduce(lambda x, y: x * y, plevres)
    return cuboid_vol, ranges


def normalize_rob(tmp_robustness):
    robustness = []
    tot_rob = reduce(lambda x, y: x + y, tmp_robustness)
    for r in tmp_robustness:
        robustness.append(r/tot_rob)
    return robustness

def main_loop(models, inp_fils, post_files):
    norm_rob = np.zeros([models, 3])
    for i in range(3):
        tmp_robustness = []
        for j in range(models):
            lims, init = read_input(inp_fils[j])
            prior_vol = calc_prior_hypercuboid(lims)
            data = read_posterior(post_files[i][j], init)
            cuboid_vol, ranges = define_post_hypercuboid(data)
            tmp_robustness.append(cuboid_vol / prior_vol)
        norm_rob[:, i] = normalize_rob(tmp_robustness)
        rob_cuboid = norm_rob.flatten()

    #models = 3
    #inp_fils = ['input_model1.xml', 'input_model2.xml', 'input_model3.xml']
    #post_files = ['results_model1/results_SIRmodel1/Population_11/data_Population11.txt',
    #              'results_model2/results_SIRmodel2/Population_10/data_Population10.txt',
    #              'results_model3/results_SIRmodel3/Population_11/data_Population11.txt']
    # index = np.arange(models)
    # tmp_robustness = []
    # for j in range(models):
    #     lims, init = read_input(inp_fils[j])
    #     prior_vol = calc_prior_hypercuboid(lims)
    #     data = read_posterior(post_files[j], init)
    #
    #
    #     robustness_cuboid = normalize_rob(tmp_robustness)

    return rob_cuboid


# colors = ['#1d91c0', '#225ea8', '#253494']
# sns.set(style="white", palette="muted", color_codes=True)
# pp = PdfPages("robustness_cuboids.pdf")
# fig, ax = plt.subplots()
# ax.bar(index, robustness, alpha=0.7, align='center',
#        color=colors)
# ax.set_xticks(index)
# ax.set_xticklabels(['1','2','3'])
# sns.despine()
# pp.savefig()
# plt.close()
# pp.close()