import networkx as nx
import numpy as np
import os

from thesis_master.src.load import load_docs
from thesis_master.src.models_implementation.model import Model

import pickle

from collections import defaultdict

from thesis_master.src.models_implementation.preproces_pipeline import subject_case_articles, preprocess

GENERATE_DISPERSION = True


class DispersionModel(Model):

    def vectorspace_and_data(self, modelname):
        disp_matrix = pickle.load(open(os.path.join(self.save_path, 'dispersion_model'), 'rb'))

        # get data
        datadump = open(self.filename, 'rb')
        data = pickle.load(datadump)
        datadump.close()

        return disp_matrix, data

    def train(self):
        if GENERATE_DISPERSION:
            # Load cases and collecting cases
            documents = load_docs(self.documents)

            subjects, cases, articles = zip(*(subject_case_articles(c) for c in documents[0]))

            # Preprocessing datasets
            cases = list(map(lambda c: preprocess(c), cases))

            data = (documents[1], subjects, cases, articles)

            datadump = open(self.filename, 'wb')
            pickle.dump(data, datadump)
            datadump.close()

        datadump = open(self.filename, 'rb')
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
        # pickle.dump((cases_graph, neigbours, G, remove), open('graph_01.jar', 'wb'))
        # Load datastructures
        # cases_graph, neigbours, G, remove = pickle.load(open('graph_05.jar', 'rb'))

        print("Graphs generated: {} nodes added".format(len(G.nodes())))

        # return cases_graph, G, neigbours, data[1], remove

        disp_matrix = np.full((len(G.nodes()), len(G.nodes())), 0, dtype=int)

        for row in range(len(G.nodes())):
            for col in range(row):
                common = neigbours[row].intersection(neigbours[col])
                disp_matrix[row][col] = len(common) * (len(common) - 1) - (
                        sum([len(neigbours[com].intersection(common)) for com in common]) / 2)

        pickle.dump(disp_matrix, open(os.path.join(self.save_path, 'dispersion_model'), 'wb'))
        return disp_matrix


root = os.path.join("models", "dispersion", "2005")
if not os.path.exists(root):
    os.makedirs(root)
year_path = "year"
year_dir = os.path.join(root, year_path)
if not os.path.exists(year_dir):
    os.mkdir(year_dir)
parseddata_path = os.path.join("data", "parsed_data", "2005")
preprocessed_data_path = os.path.join(year_dir, "dispersion_cases_preprocessed.data")
dm = DispersionModel(
    preprocessed_data_path,
    year_dir,
    parseddata_path
)
# dm.train()
score_path = os.path.join(year_dir, "dispersion_scores")
dm.full_run("dispersion_model", reduced=True, dimensions=2, filepath=score_path)
