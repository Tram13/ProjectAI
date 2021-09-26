import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from thesis_master.src.models_implementation.color_distribution import legend
from thesis_master.src.models_implementation.preproces_pipeline import preprocess, subject_case_articles
from thesis_master.src.load import load_docs

from collections import defaultdict
import progressbar
import pickle

from sklearn.manifold import TSNE
from sklearn import decomposition

from thesis_master.src.models_implementation.jqm_cvi import dunn, dunn_fast
from collections import defaultdict
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.patches as mpatches
import matplotlib.figure as figure
from s_dbw import S_Dbw


def generate_graph(documents):
    # Load cases and collecting cases
    # documents = load_docs(documents)

    # subjects, cases, articles = zip(*(subject_case_articles(c) for c in documents[0]))

    # Preprocessing datasets
    # cases = list(map(lambda c: preprocess(c), progressbar.progressbar(cases, redirect_stdout=True)))

    # data = (documents[1], subjects, cases, articles)

    # datadump = open('data_05.jar', 'wb')
    # pickle.dump(data, datadump)
    # datadump.close()

    datadump = open('data_05.jar', 'rb')
    data = pickle.load(datadump)
    datadump.close()

    # define and fill articles-dict
    articles_dict = defaultdict(list)
    cases_graph = np.full((len(data[0]), len(data[0])), 0, dtype=int)
    neigbours = [set() for _ in range(len(data[0]))]

    # parse articles into dictst
    for i, articles in enumerate(data[3]):
        for art in articles.split("||"):
            if not art == ' ':
                articles_dict[art].append(i)
            for link in articles_dict[art]:
                # Update weight of edge i-link
                cases_graph[i][link] += 1
                cases_graph[link][i] += 1

                # Update neighbours
                neigbours[i].add(link)
                neigbours[link].add(i)

    print("Dictionaries ready")

    G = nx.Graph()
    G.add_nodes_from(data[0])

    for n1 in range(len(cases_graph)):
        for n2 in range(n1):
            if cases_graph[n1][n2] > 0:
                G.add_edge(data[0][n1], data[0][n2], weight=cases_graph[n1][n2])

    remove = [data[0].index(node) for node, degree in dict(G.degree()).items() if degree < 2]
    removing = [node for node, degree in dict(G.degree()).items() if degree < 2]
    #
    G.remove_nodes_from(removing)

    print("Networkx generated")

    # Save datastructures
    pickle.dump((cases_graph, neigbours, G, remove), open('graph_01.jar', 'wb'))
    # Load datastructures
    # cases_graph, neigbours, G, remove = pickle.load(open('graph_05.jar', 'rb'))

    print("Graphs generated: {} nodes added".format(len(G.nodes())))

    return cases_graph, G, neigbours, data[1], remove


def plot_graph(documents):
    _, G, _, documents, remove = generate_graph(documents)

    datadump = open('data_01.jar', 'rb')
    data = pickle.load(datadump)
    datadump.close()

    print("Data fetched")

    # fixing the size of the figure
    plt.figure(figsize=(15, 15))

    subjects = [data[1][i] for i in range(len(data[0])) if i not in remove]

    print(len(subjects))

    # node colour is a list of degrees of nodes
    node_color = [legend[subject] for subject in subjects]

    # width of edge is a list of weight of edges
    edge_width = [0.1 * G[u][v]['weight'] for u, v in G.edges()]

    nx.draw_kamada_kawai(G, alpha=0.7,
                         node_color=node_color,
                         edge_width=edge_width,
                         with_labels=False)

    # plt.axis('off')
    # plt.tight_layout()
    plt.savefig('../plots/full_jap_01.png')
    plt.show()


def calculate_dispersion(documents):
    # Load graph data
    _, G, neigbours, documents, remove = generate_graph(documents)
    print("Data fetched.")

    disp_matrix = np.full((len(G.nodes()), len(G.nodes())), 0, dtype=int)

    for row in range(len(G.nodes())):
        for col in range(row):
            common = neigbours[row].intersection(neigbours[col])
            disp_matrix[row][col] = len(common) * (len(common) - 1) - (
                        sum([len(neigbours[com].intersection(common)) for com in common]) / 2)

    return disp_matrix


def calculate_embeddedness(documents):
    # Load graph data
    _, G, neigbours, documents, remove = generate_graph(documents)
    print("Data fetched.")

    emb_matrix = np.full((len(G.nodes()), len(G.nodes())), 0, dtype=int)

    for row in range(len(G.nodes())):
        for col in range(row):
            emb_matrix[row][col] = len(neigbours[row].intersection(neigbours[col]))

    return emb_matrix


# disp = calculate_dispersion("../extensive_parsed/2005")
# pickle.dump(disp, open('dis05.jar', 'wb'))
dis = pickle.load(open('dis05.jar', 'rb'))
# emb = pickle.load(open('emb05.jar', 'rb'))
graph, G, _, documents, remove = generate_graph("../extensive_parsed_2005")

# reduced = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=600).fit_transform(dis)
# reduced = decomposition.PCA(n_components=20).fit_transform(emb)
# reduced = dis

####### Plotting #######
# fig, ax = plt.subplots()
#
# fig.set_size_inches(8, 6, forward=True)

subjects = [documents[i] for i in range(len(documents)) if i not in remove]

# for (x, y), group in zip(reduced, documents):
#     ax.scatter(x, y, label=group, color=legend[group], alpha=0.6, edgecolors='none')
#
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 - box.height * 0.001,
#                  box.width, box.height * 0.9])

# plt.show()
#
# fig.savefig("../plots/snek_dis_pca.png", dpi=100)

####### Silhouette #######
# print("silhouette : {}".format(silhouette_score(reduced, subjects)))

####### Mean intracluster distance #######
# totaldict = defaultdict(list)
# for i, vector in enumerate(reduced):
#     totaldict[subjects[i]].append(vector)
# maxim = np.max(euclidean_distances(reduced))

# means = [( label, np.mean(euclidean_distances(totaldict[label]))/maxim) for label in totaldict.keys()]
# print("Mean intracluster distance: {}".format(np.mean([x[1] for x in means]), means))

####### dunn_index #######
# sub = ['Bestuursrecht', 'Bestuursrecht; Ambtenarenrecht',
#  'Bestuursrecht; Belastingrecht', 'Bestuursrecht; Bestuursstrafrecht',
#  'Bestuursrecht; Europees bestuursrecht', 'Bestuursrecht; Omgevingsrecht',
#  'Bestuursrecht; Socialezekerheidsrecht',
#  'Bestuursrecht; Vreemdelingenrecht', 'Civiel recht',
#  'Civiel recht; Insolventierecht',
#  'Civiel recht; Personen- en familierecht',
#  'Civiel recht; Verbintenissenrecht', 'Strafrecht']
# labels = [sub.index(dat) for dat in subjects]
# print("dunn index: {}".format(dunn_fast(reduced, labels)))

# davies_boulding
# print("davies_boulding: {}".format(davies_bouldin_score(reduced, subjects)))

####### S_Dbw #######
# print("S_Dbw: {}".format(S_Dbw(reduced, labels)))

terms = ['Bestuursrecht', 'Civiel recht', 'Strafrecht', 'Bestuursrecht; Belastingrecht',
         'Bestuursrecht; Socialezekerheidsrecht', 'Bestuursrecht; Vreemdelingenrecht',
         'Civiel recht; Personen- en familierecht', 'Bestuursrecht; Omgevingsrecht',
         'Bestuursrecht; Ambtenarenrecht',
         'Civiel recht; Insolventierecht']

n = len(terms)

cluster_labels = ['' for i in range(n)]

picked = [i for i in range(n)]

clusters = KMeans(n_clusters=n).fit_predict(dis)

high_score = 0

for subject in terms:
    position = max(
        [(len([1 for j in range(len(clusters)) if (clusters[j] == i and subjects[j] == subject)]), i) for i in picked if
         i is not -1], key=lambda x: x[0])
    picked[position[1]] = -1
    cluster_labels[position[1]] = subject
    high_score += position[0]

print(high_score / len(subjects))

fig, ax = plt.subplots()

fig.set_size_inches(8, 6, forward=True)

reduced = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=600).fit_transform(dis)

new_labels = [cluster_labels[label] for label in clusters]

for (x, y), group in zip(reduced, new_labels):
    ax.scatter(x, y, label=group, color=legend[group], alpha=0.6, edgecolors='none')

box = ax.get_position()
ax.set_position([box.x0, box.y0 - box.height * 0.001,
                 box.width, box.height * 0.9])

plt.show()
