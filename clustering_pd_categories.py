import gensim
import os
import pandas as pd

from src.data_extraction import DataExtractionPDTexts
from src.vectorizer import Vectorizer
from src.clusters_data_saver import ClustersDataPDCategories
from src.clusterizer import Clusterizer


def main():
    project_path = os.getcwd()
    model_path = rf'{project_path}\models\geowac\model.model'
    geowac_model = gensim.models.KeyedVectors.load(model_path)
    extractor = DataExtractionPDTexts(rf'{project_path}\data\preprocessed_cleaned_pd.xlsx')
    extractor.category_types = ['clean_animals']
    vectoriser = Vectorizer(geowac_model)
    cluster_saver = ClustersDataPDCategories(extractor, geowac_model)
    clusters_getter = Clusterizer(geowac_model)

    for page in ['healthy', 'PD']:

        for category in extractor.category_types:
            print(f'Clustering {category} category in {page}')
            sequence_series = extractor.get_series(page, category)  # getting words lists from a column
            clusters_list = []  # a list of lists of clusters for current column

            for words_string in sequence_series:
                if not isinstance(words_string, str):  # dealing with NaNs or other non-string values
                    clusters_list.append([])
                    continue

                tokens_sequence = vectoriser.get_sequence(words_string)
                # string of words coverted to list with special tags

                cell_clusters = clusters_getter.cluster(tokens_sequence)
                clusters_list.append(cell_clusters)

            cluster_saver.add_column(page, category,
                                     pd.Series(clusters_list))
            # adding clusters column in a table

            # counting metrics
            cluster_saver.count_num_switches(page, category)
            cluster_saver.count_mean_cluster_size(page, category)
            cluster_saver.count_mean_distances(page, category)
            cluster_saver.count_mean_silhouette_score(page, category)
            cluster_saver.count_cluster_t_scores(page, category)

    print('Finishing clustering!')
    # saving
    cluster_saver.save_excel(rf'{project_path}\result\pd_categories\clusters_metrics_dataset_animals.xlsx')


if __name__ == '__main__':
    main()
