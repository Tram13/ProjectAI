import pickle

import numpy as np
import os

from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.figure as figure
from thesis_master.src.models_implementation.color_distribution import legend

import itertools

from collections import defaultdict
from abc import ABC, abstractmethod

from thesis_master.src.models_implementation.jqm_cvi import dunn, dunn_fast
from s_dbw import S_Dbw


class Model(ABC):

    def __init__(self, filename, save_path, documents):
        self.filename = filename
        self.save_path = save_path
        self.documents = documents

    # abstract method
    def train(self):
        pass

    # abstract method
    def query(self, case, modelname):
        pass

    # abstract method
    def vectorspace_and_data(self, modelname):
        pass

    def plot_vectorspace(self, name, decomp, modelname):
        vectorspace, data = self.vectorspace_and_data(modelname)
        # reduce the vectorspaces to get a 2d representation
        if decomp == "pca":
            decomposer = decomposition.PCA(n_components=2)
        elif decomp == "tsne":
            decomposer = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

        reduced = decomposer.fit_transform(vectorspace)

        fig, ax = plt.subplots()

        fig.set_size_inches(8, 6, forward=True)

        for (x, y), group in zip(reduced, data[1]):
            ax.scatter(x, y, label=group, color=legend[group], alpha=0.6, edgecolors='none')

        # Put a legend below current axis
        # ax.legend(handles=[mpatches.Patch(color=legend[key], label=key) for key in legend.keys()], loc='upper right', bbox_to_anchor=(1.5, 1),
        #       fancybox=True, shadow=True, ncol=1, prop={'size': 20})

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - box.height * 0.001,
                         box.width, box.height * 0.9])

        plt.show()
        # fig.savefig("../../plots/{}_1_{}.png".format(name, decomp), dpi=100)

    def silhouette_score(self, modelname, reduced=False, dimensions=2):
        vectorspace, data = self.vectorspace_and_data(modelname)

        if reduced:
            if dimensions == 2:
                vectorspace = TSNE(n_components=dimensions, verbose=1, perplexity=40, n_iter=300).fit_transform(
                    vectorspace)
            else:
                vectorspace = decomposition.PCA(n_components=dimensions).fit_transform(
                    vectorspace)

        return silhouette_score(vectorspace, data[1])

    def mean_intracluster_distance(self, modelname, reduced=False, dimensions=2):
        vectorspace, data = self.vectorspace_and_data(modelname)

        if reduced:
            if dimensions == 2:
                vectorspace = TSNE(n_components=dimensions, verbose=1, perplexity=40, n_iter=300).fit_transform(
                    vectorspace)
            else:
                vectorspace = decomposition.PCA(n_components=dimensions).fit_transform(
                    vectorspace)

        totaldict = defaultdict(list)
        for i, vector in enumerate(vectorspace):
            totaldict[data[1][i]].append(vector)
        maxim = np.max(euclidean_distances(vectorspace))

        means = [(label, np.mean(euclidean_distances(totaldict[label])) / maxim) for label in totaldict.keys()]
        return np.mean([x[1] for x in means]), means

    def dunn_index(self, modelname, reduced=False, dimensions=2):

        subjects = ['Bestuursrecht', 'Bestuursrecht; Ambtenarenrecht',
                    'Bestuursrecht; Belastingrecht', 'Bestuursrecht; Bestuursstrafrecht',
                    'Bestuursrecht; Europees bestuursrecht', 'Bestuursrecht; Omgevingsrecht',
                    'Bestuursrecht; Socialezekerheidsrecht',
                    'Bestuursrecht; Vreemdelingenrecht', 'Civiel recht',
                    'Civiel recht; Insolventierecht',
                    'Civiel recht; Personen- en familierecht',
                    'Civiel recht; Verbintenissenrecht', 'Strafrecht']

        vectorspace, data = self.vectorspace_and_data(modelname)

        if reduced:
            if dimensions == 2:
                vectorspace = TSNE(n_components=dimensions, verbose=1, perplexity=40, n_iter=300).fit_transform(
                    vectorspace)
            else:
                vectorspace = decomposition.PCA(n_components=dimensions).fit_transform(
                    vectorspace)

        labels = [subjects.index(dat) for dat in data[1]]
        return dunn_fast(vectorspace, labels)

    def davies_boulding(self, modelname, reduced=False, dimensions=2):
        vectorspace, data = self.vectorspace_and_data(modelname)

        if reduced:
            if dimensions == 2:
                vectorspace = TSNE(n_components=dimensions, verbose=1, perplexity=40, n_iter=300).fit_transform(
                    vectorspace)
            else:
                vectorspace = decomposition.PCA(n_components=dimensions).fit_transform(
                    vectorspace)

        return davies_bouldin_score(vectorspace, data[1])

    def full_run(self, modelname, reduced=False, dimensions=None, filepath=''):
        silhouette_score_ = self.silhouette_score(modelname, reduced=reduced, dimensions=dimensions)
        dunn_score_ = self.dunn_index(modelname, reduced=reduced, dimensions=dimensions)
        davies_bouldin_score_ = self.davies_boulding(modelname, reduced=reduced, dimensions=dimensions)
        mean_intracluster_distance_ = self.mean_intracluster_distance(modelname, reduced=reduced, dimensions=dimensions)
        s_dbw_ = self.s_dbw_validity(modelname, reduced=reduced, dimensions=dimensions)

        silhouette_score_string = "silhouette score: {}".format(silhouette_score_)
        dunn_score_string = "dunn score: {}".format(dunn_score_)
        davies_bouldin_score_string = "davies-bouldin score: {}".format(davies_bouldin_score_)
        mean_intracluster_distance_string = "mean intracluster distance: {}".format(mean_intracluster_distance_)
        s_dbw_string = "S_Dbw: {}".format(s_dbw_)

        # print(silhouette_score_string)
        # print(dunn_score_string)
        # print(davies_bouldin_score_string)
        # print(mean_intracluster_distance_string)
        # print(s_dbw_string)

        # make sure we do not append
        if os.path.exists(filepath):
            os.remove(filepath)

        dictionary_data = {"silhouette": silhouette_score_,
                           "dunn": dunn_score_,
                           "davies-bouldin": davies_bouldin_score_,
                           "mean intracluster distance": mean_intracluster_distance_,
                           "S_Dbw": s_dbw_
                           }
        a_file = open(filepath, "wb")
        pickle.dump(dictionary_data, a_file)
        a_file.close()

        # how to read the dictionary:
        # a_file = open("filename.pkl", "rb")
        # output_dictionary = pickle.load(a_file)
        # a_file.close()

    def s_dbw_validity(self, modelname, reduced=False, dimensions=2):
        vectorspace, data = self.vectorspace_and_data(modelname)

        subjects = ['Bestuursrecht', 'Bestuursrecht; Ambtenarenrecht',
                    'Bestuursrecht; Belastingrecht', 'Bestuursrecht; Bestuursstrafrecht',
                    'Bestuursrecht; Europees bestuursrecht', 'Bestuursrecht; Omgevingsrecht',
                    'Bestuursrecht; Socialezekerheidsrecht',
                    'Bestuursrecht; Vreemdelingenrecht', 'Civiel recht',
                    'Civiel recht; Insolventierecht',
                    'Civiel recht; Personen- en familierecht',
                    'Civiel recht; Verbintenissenrecht', 'Strafrecht']

        if reduced:
            if dimensions == 2:
                vectorspace = TSNE(n_components=dimensions, verbose=1, perplexity=40, n_iter=300).fit_transform(
                    vectorspace)
            else:
                vectorspace = decomposition.PCA(n_components=dimensions).fit_transform(
                    vectorspace)

        labels = [subjects.index(dat) for dat in data[1]]

        return S_Dbw(vectorspace, labels)

    def cluster_analysis(self, modelname):
        vectorspace, data = self.vectorspace_and_data(modelname)

        subjects = ['Bestuursrecht', 'Civiel recht', 'Strafrecht', 'Bestuursrecht; Belastingrecht',
                    'Bestuursrecht; Socialezekerheidsrecht', 'Bestuursrecht; Vreemdelingenrecht',
                    'Civiel recht; Personen- en familierecht', 'Bestuursrecht; Omgevingsrecht',
                    'Bestuursrecht; Ambtenarenrecht',
                    'Civiel recht; Insolventierecht', 'Civiel recht; Verbintenissenrecht',
                    'Bestuursrecht; Europees bestuursrecht', 'Bestuursrecht; Bestuursstrafrecht']

        n = len(subjects)

        cluster_labels = ['' for i in range(n)]

        picked = [i for i in range(n)]

        clusters = KMeans(n_clusters=n).fit_predict(vectorspace)

        high_score = 0

        for subject in subjects:
            position = max(
                [(len([1 for j in range(len(clusters)) if clusters[j] == i and data[1][j] == subject]), i) for i in
                 picked if i != -1], key=lambda x: x[0])
            picked[position[1]] = -1
            cluster_labels[position[1]] = subject
            high_score += position[0]

        fig, ax = plt.subplots()

        fig.set_size_inches(8, 6, forward=True)

        reduced = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=600).fit_transform(vectorspace)

        new_labels = [cluster_labels[label] for label in clusters]

        for (x, y), group in zip(reduced, new_labels):
            ax.scatter(x, y, label=group, color=legend[group], alpha=0.6, edgecolors='none')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - box.height * 0.001,
                         box.width, box.height * 0.9])

        return high_score / len(vectorspace)
