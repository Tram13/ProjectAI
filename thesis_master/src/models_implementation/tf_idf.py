from datetime import datetime

import joblib
import os
from numpy import mean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from thesis_master.src.models_implementation.model import Model
from thesis_master.src.models_implementation.preproces_pipeline import preprocess, subject_case_summary
from thesis_master.src.load import load_docs

import pickle

GENERATE_TFIDF = True


def tokenize(x):
    return x


class TFIDFModel(Model):

    def train(self):

        if GENERATE_TFIDF:
            # Load cases and collecting cases
            documents = load_docs(self.documents)

            subjects, cases, summaries = zip(*(subject_case_summary(c) for c in documents[0]))

            # Preprocessing datasets
            cases = list(map(lambda c: preprocess(c), cases))

            data = (documents[1], subjects, cases, summaries)

            datadump = open(self.filename, 'wb')
            pickle.dump(data, datadump)
            datadump.close()

        datadump = open(self.filename, 'rb')
        data = pickle.load(datadump)
        datadump.close()

        print('All preprocessing done, fitting matrix')

        tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 1), tokenizer=tokenize, lowercase=False)
        tfidf_matrix = tfidf.fit_transform(data[2])

        joblib.dump((tfidf, tfidf_matrix), self.save_path + "tf_idf_model")

    def query(self, case, modelname):
        # loading model
        tfidf, tfidf_matrix = joblib.load(self.save_path + '/' + modelname)

        # load data
        datadump = open(self.filename, 'rb')
        data = pickle.load(datadump)
        datadump.close()

        case = tfidf.transform([preprocess(case)])
        similarity = cosine_similarity(case, tfidf_matrix)[0]

        print("Similarity Score ", similarity)
        print("Best match: {} with similarity {}".format(data[0][max(enumerate(similarity), key=lambda x: x[1])[0]],
                                                         max(enumerate(similarity), key=lambda x: x[1])))
        return [(data[0][i[0]], data[3][i[0]]) for i in sorted(enumerate(similarity), key=lambda x: x[1])[:10]]

    def vectorspace_and_data(self, modelname):
        # loading model
        tfidf, tfidf_matrix = joblib.load(self.save_path + modelname)

        # get data
        datadump = open(self.filename, 'rb')
        data = pickle.load(datadump)
        datadump.close()

        # get the vectorspaces
        docvecs = tfidf_matrix.toarray()

        return docvecs, data


root = os.path.join("models", "tf_idf", "2005")
if not os.path.exists(root):
    os.makedirs(root)
year_path = "year"
year_dir = os.path.join(root, year_path)
if not os.path.exists(year_dir):
    os.mkdir(year_dir)
parseddata_path = os.path.join("data", "parsed_data", "2005")
preprocessed_data_path = os.path.join(year_dir, "tf_idf_cases_preprocessed.data")
dm = TFIDFModel(
    preprocessed_data_path,
    year_dir,
    parseddata_path
)
dm.train()
score_path = os.path.join(year_dir, "tf_idf_scores")
dm.full_run("tf_idf_model", reduced=True, dimensions=2, filepath=score_path)