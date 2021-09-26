import pickle

import networkx as nx

cases_graph, neigbours, G, remove = pickle.load(open('graph_05.jar', 'rb'))
dispersion = nx.dispersion(G, normalized=False)

pickle.dump(dispersion, open('dispersion.jar', 'wb'))
