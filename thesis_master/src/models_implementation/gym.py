from thesis_master.src.models_implementation.tf_idf import TFIDFModel

tfidf = TFIDFModel(
    "jar/tfidf_cases2005.data",
    "../../models/leegle/",
    "../../simple_parsed_2005"
)


tfidf.train()
# tfidf.query("Bestuursrecht", "tfidf_unigram")
# tfidf.full_run("tfidf_20200522-001819", reduced=True, dimensions=2)