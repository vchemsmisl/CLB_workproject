import gensim
import pandas as pd

from src.data_extraction import DataExtractionSchizophrenia
from src.vectorizer import Vectorizer
from src.clusters_data_saver import ClustersDataSchizophrenia
from src.clusterizer import Clusterizer


def main():
    model_path = 'C:\programming\CLB_workproject\models\geowac\model.model'
    data_path = 'C:\programming\CLB_workproject\data\schizophrenia_data\schiz_transcripts_preprocessed_clean.xlsx'
    clustered_data_path = 'C:\programming\CLB_workproject\\result\schizophrenia\schiz_clusters_clean.xlsx'

    geowac_model = gensim.models.KeyedVectors.load(model_path)
    extractor = DataExtractionSchizophrenia(data_path)
    vectoriser = Vectorizer(geowac_model)
    cluster_saver = ClustersDataSchizophrenia(extractor, geowac_model)
    clusters_getter = Clusterizer(geowac_model)

    # general principle: clusterising one cell at a time
    for category in ['action', 'fruit', 'instrument']:
        sequence_series = extractor.get_series(category)  # getting words lists from a column
        clusters_list = []  # a list of lists of clusters for current column

        for words_string in sequence_series:
            if not isinstance(words_string, str):  # dealing with NaNs or other non-string values
                clusters_list.append([])
                continue

            tokens_sequence = vectoriser.get_sequence(words_string)  # string of words coverted to list with special tags
            vectoriser.update_dict(words_string)  # adding words embeddings to a dict
            cell_clusters = clusters_getter.cluster(tokens_sequence)  # converting list of words to list of clusters
            clusters_list.append(cell_clusters)

        cluster_saver.add_column(category, pd.Series(clusters_list))  # adding clusters column in a table

        cluster_saver.count_num_switches(category)
        cluster_saver.count_mean_cluster_size(category)
        cluster_saver.count_mean_distances(category)
        cluster_saver.count_mean_silhouette_score(category)
        cluster_saver.count_cluster_t_scores(category)

        # vectoriser.update_json()  # updating a json-file with words embeddings from a dict for each category

    cluster_saver.save_excel(clustered_data_path)


if __name__ == '__main__':
    main()
