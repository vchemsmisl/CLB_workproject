import pandas as pd
import gensim
import numpy as np
from itertools import permutations
from src.data_extraction import DataExtraction


class ClustersDataBase:
    pass


class ClustersDataAphasia(ClustersDataBase):
    def __init__(self, extractor: DataExtraction, model: gensim.models.fasttext.FastTextKeyedVectors) -> None:
        self.extractor = extractor
        self.id_healthy = extractor.get_ids()
        self.id_aphasia = extractor.get_ids('aphasia')
        self.healthy_data = pd.DataFrame(self.id_healthy)
        self.aphasia_data = pd.DataFrame(self.id_aphasia)
        self.model = model

    def get_df(self, sheet):
        if sheet == 'healthy':
            return self.healthy_data
        return self.aphasia_data

    @staticmethod
    def get_column_name(category: str, lexemes: str) -> str:
        category_types = {'animals':'a',
                          'professions':'b',
                          'cities':'c'}
        return f'C6({category_types.get(category)})-{lexemes}'

    @staticmethod
    def avg_cluster_size(row: pd.Series) -> float:
        """
        Get average cluster size in a row
        """
        clusters_sizes = []
        for cell in row:
            clusters_sizes.extend(len(cluster) for cluster in cell)
        return sum(clusters_sizes) / len(clusters_sizes)

    def avg_cluster_distance(self, cluster_sequence):
        """
        Count average cluster distance
        """
        if not cluster_sequence:
            return np.NaN

        centroids_dict = {}
        distances = []

        for cluster in cluster_sequence:
          centroid = sum(self.model[word] for word in cluster) / len(cluster)
          centroids_dict[tuple(cluster)] = centroid

        for idx in range(0, len(cluster_sequence)-1):
            cluster_1 = cluster_sequence[idx]
            cluster_2 = cluster_sequence[idx+1]
            Dij = np.dot(gensim.matutils.unitvec(centroids_dict[tuple(cluster_1)]),
                              gensim.matutils.unitvec(centroids_dict[tuple(cluster_2)]))
            distances.append(Dij)

        if not distances:
            return np.NaN

        return sum(distances)/len(distances)

    def count_distances(self, sheet_name):
        """
        Counting distances for all columns
        """
        clean_columns = ['Mean_distance_animals_clean',
                         'Mean_distance_professions_clean',
                         'Mean_distance_cities_clean']

        all_columns = ['Mean_distance_animals_all',
                       'Mean_distance_professions_all',
                       'Mean_distance_cities_all']
        if sheet_name == 'healthy':
            location = 0
            for column in ['animals', 'professions', 'cities']:
                location += 3
                self.healthy_data.insert(loc = location,
                                    column = f'Mean_distance_{column}_clean',
                                    value = self.healthy_data[self.get_column_name(column, 'clean')].apply(self.avg_cluster_distance))


            self.healthy_data.insert(loc = location + 1,
                                   column = 'Average_distances_clean',
                                   value = self.healthy_data[clean_columns].mean(axis=1))

            location = 10

            for column in ['animals', 'professions', 'cities']:
                location += 3
                self.healthy_data.insert(loc = location,
                                    column = f'Mean_distance_{column}_all',
                                    value = self.healthy_data[self.get_column_name(column, 'clean-all-lexemes')].apply(self.avg_cluster_distance))


            self.healthy_data.insert(loc = location + 1,
                                   column = 'Average_distances_all',
                                   value = self.healthy_data[all_columns].mean(axis=1))

        else:
            location = 0
            for column in ['animals', 'professions', 'cities']:
                location += 3
                self.aphasia_data.insert(loc = location,
                                    column = f'Mean_distance_{column}_clean',
                                    value = self.aphasia_data[self.get_column_name(column, 'clean')].apply(self.avg_cluster_distance))

            self.aphasia_data.insert(loc = location + 1,
                                   column = 'Average_distances_clean',
                                   value = self.aphasia_data[clean_columns].mean(axis=1))

            location = 10
            for column in ['animals', 'professions', 'cities']:
                location += 3
                self.aphasia_data.insert(loc = location,
                                    column = f'Mean_distance_{column}_all',
                                    value = self.aphasia_data[self.get_column_name(column, 'clean-all-lexemes')].apply(self.avg_cluster_distance))

            self.aphasia_data.insert(loc = location + 1,
                               column = 'Average_distances_all',
                               value = self.aphasia_data[all_columns].mean(axis=1))

    def silhouette_score(self, cluster_sequence):
        silhouette_coefs = []

        for idx, cluster in enumerate(cluster_sequence):
            for word_1 in cluster:

                a = sum(self.model.similarity(word_1, word_2)
                    for word_2 in cluster if word_1 != word_2) / len(cluster)

                if idx != len(cluster_sequence) - 1:
                    b = sum(self.model.similarity(word_1, word_2)
                    for word_2 in cluster_sequence[idx + 1]) / len(cluster_sequence[idx + 1])
                else:
                    b = sum(self.model.similarity(word_1, word_2)
                    for word_2 in cluster_sequence[idx - 1]) / len(cluster_sequence[idx - 1])

                s = (b - a) / max(a, b)
                silhouette_coefs.append(s)

        if silhouette_coefs:
            return sum(silhouette_coefs) / len(silhouette_coefs)
        return np.NaN

    def count_silhouette(self, sheet_name):
        """
        Counting distances for all columns
        """
        clean_columns = ['Silhouette_score_animals_clean',
                      'Silhouette_score_professions_clean',
                      'Silhouette_score_cities_clean']

        all_columns = ['Silhouette_score_animals_all',
                    'Silhouette_score_professions_all',
                    'Silhouette_score_cities_all']
        if sheet_name == 'healthy':
            location = 0
            for column in ['animals', 'professions', 'cities']:
                location += 3
                self.healthy_data.insert(loc = location,
                                  column = f'Silhouette_score_{column}_clean',
                                  value = self.healthy_data[self.get_column_name(column, 'clean')].apply(self.silhouette_score))


            self.healthy_data.insert(loc = location + 1,
                                column = 'Average_silhouette_score__clean',
                                value = self.healthy_data[clean_columns].mean(axis=1))

            location = 10

            for column in ['animals', 'professions', 'cities']:
                location += 3
                self.healthy_data.insert(loc = location,
                                  column = f'Silhouette_score_{column}_all',
                                  value = self.healthy_data[self.get_column_name(column, 'clean-all-lexemes')].apply(self.silhouette_score))


            self.healthy_data.insert(loc = location + 1,
                                column = 'Average_silhouette_score_all',
                                value = self.healthy_data[all_columns].mean(axis=1))

        else:
            location = 0
            for column in ['animals', 'professions', 'cities']:
                location += 3
                self.aphasia_data.insert(loc = location,
                                  column = f'Silhouette_score_{column}_clean',
                                  value = self.aphasia_data[self.get_column_name(column, 'clean')].apply(self.silhouette_score))

            self.aphasia_data.insert(loc = location + 1,
                                column = 'Average_silhouette_score_clean',
                                value = self.aphasia_data[clean_columns].mean(axis=1))

            location = 10
            for column in ['animals', 'professions', 'cities']:
                location += 3
                self.aphasia_data.insert(loc = location,
                                  column = f'Silhouette_score_{column}_all',
                                  value = self.aphasia_data[self.get_column_name(column, 'clean-all-lexemes')].apply(self.silhouette_score))

            self.aphasia_data.insert(loc = location + 1,
                                column = 'Average_silhouette_score__all',
                                value = self.aphasia_data[all_columns].mean(axis=1))

    @staticmethod
    def cluster_t_score(f_n, f_c, f_nc, N):
        if f_nc == 0:
            return 0
        numerator = f_nc - f_n * f_c / N
        denominator = np.sqrt(f_nc)
        return numerator / denominator


    def avg_cluster_t_score(self, cell, column_clusters):
        all_words = ' '.join([word for cell in column_clusters for cluster in cell for word in cluster])
        N = len(all_words)

        cell_t_scores = []
        for cluster in cell:
            all_wordpairs = list(permutations(cluster, 2))

            pairwise_t_scores = []
            for wordpair in all_wordpairs:
                f_n = all_words.count(wordpair[0])
                f_c = all_words.count(wordpair[1])
                f_nc = all_words.count(' '.join((wordpair[0], wordpair[1])))
                f_nc += all_words.count(' '.join((wordpair[1], wordpair[0])))

                t_score = self.cluster_t_score(f_n, f_c, f_nc, N)
                pairwise_t_scores.append(t_score)
            cell_t_scores.extend(pairwise_t_scores)

        return sum(cell_t_scores)

    def count_cluster_t_scores(self, sheet_name):
        clean_columns = ['Mean_cluster_t_score_animals_clean',
                     'Mean_cluster_t_score_professions_clean',
                     'Mean_cluster_t_score_cities_clean']

        all_columns = ['Mean_cluster_t_score_animals_all',
                   'Mean_cluster_t_score_professions_all',
                   'Mean_cluster_t_score_cities_all']
        if sheet_name == 'healthy':
            location = 0
            for column in ['animals', 'professions', 'cities']:
                location += 4
                self.healthy_data.insert(loc = location,
                                column = f'Mean_cluster_t_score_{column}_clean',
                                value = self.healthy_data[self.get_column_name(column, 'clean')].apply(
                                    lambda x: self.avg_cluster_t_score(x, self.healthy_data[self.get_column_name(column, 'clean')])
                                ))

            # location += 3

            self.healthy_data.insert(loc = location + 2,
                               column = 'Average_cluster_t_score_clean',
                               value = self.healthy_data[clean_columns].mean(axis=1))

            location = 16

            for column in ['animals', 'professions', 'cities']:
                location += 4
                self.healthy_data.insert(loc = location,
                                column = f'Mean_cluster_t_score_{column}_all',
                                value = self.healthy_data[self.get_column_name(column, 'clean-all-lexemes')].apply(
                                    lambda x: self.avg_cluster_t_score(x, self.healthy_data[self.get_column_name(column, 'clean')])
                                ))

            self.healthy_data.insert(loc = location + 2,
                               column = 'Average_cluster_t_score_all',
                               value = self.healthy_data[all_columns].mean(axis=1))

        else:
            location = 0
            for column in ['animals', 'professions', 'cities']:
                location += 4
                self.aphasia_data.insert(loc = location,
                                column = f'Mean_cluster_t_score_{column}_clean',
                                value = self.aphasia_data[self.get_column_name(column, 'clean')].apply(
                                    lambda x: self.avg_cluster_t_score(x, self.aphasia_data[self.get_column_name(column, 'clean')])
                                ))

            self.aphasia_data.insert(loc = location + 2,
                               column = 'Average_cluster_t_score_clean',
                               value = self.aphasia_data[clean_columns].mean(axis=1))

            location = 16
            for column in ['animals', 'professions', 'cities']:
                location += 4
                self.aphasia_data.insert(loc = location,
                                column = f'Mean_cluster_t_score_{column}_all',
                                value = self.aphasia_data[self.get_column_name(column, 'clean-all-lexemes')].apply(
                                    lambda x: self.avg_cluster_t_score(x, self.aphasia_data[self.get_column_name(column, 'clean')])
                                ))

            self.aphasia_data.insert(loc = location + 2,
                               column = 'Average_cluster_t_score_all',
                               value = self.aphasia_data[all_columns].mean(axis=1))

    def add_column(self,
                 sheet_name: str,
                 category: str,
                 lexemes: str,
                 clusters: pd.Series) -> None:
        """
        Adding a column with clusters
        """
        if sheet_name == 'healthy':
            column_name = self.get_column_name(category, lexemes)
            self.healthy_data[column_name] = clusters

        else:
            column_name = self.get_column_name(category, lexemes)
            self.aphasia_data[column_name] = clusters

    def count_switches(self,
                     sheet_name: str,
                     category: str,
                     lexemes: str) -> None:
        """
        Count number of switches for each cell
        """
        if sheet_name == 'healthy':
            column = self.get_column_name(category, lexemes)
            new_column_name = f'Switch_number_{category}_{lexemes}'

            self.healthy_data[new_column_name] = self.healthy_data[column].apply(lambda x: len(x) - 1)

        else:
            column = self.get_column_name(category, lexemes)
            new_column_name = f'Switch_number_{category}_{lexemes}'

            self.aphasia_data[new_column_name] = self.aphasia_data[column].apply(lambda x: len(x) - 1)

    def count_mean(self, sheet_name: str) -> None:
        """
        Count mean cluster size for each row
        """
        clean_columns = [self.get_column_name(category, 'clean') for category in ['animals', 'professions', 'cities']]
        all_columns = [self.get_column_name(category, 'clean-all-lexemes') for category in ['animals', 'professions', 'cities']]

        if sheet_name == 'healthy':
            self.healthy_data.insert(loc = 11,
                                  column = 'Mean_cluster_size_clean',
                                  value = self.healthy_data[clean_columns].apply(self.avg_cluster_size, axis=1))
            self.healthy_data.insert(loc = 21,
                                  column = 'Mean_cluster_size_all',
                                  value = self.healthy_data[all_columns].apply(self.avg_cluster_size, axis=1))

        else:
            self.aphasia_data.insert(loc = 10,
                            column = 'Mean_cluster_size_clean',
                            value = self.aphasia_data[clean_columns].apply(self.avg_cluster_size, axis=1))
            self.aphasia_data.insert(loc = 20,
                                  column = 'Mean_cluster_size_all',
                                  value = self.aphasia_data[all_columns].apply(self.avg_cluster_size, axis=1))

    def save_excel(self) -> None:
        """
        Saving data with clusters to an Excel file
        """
        with pd.ExcelWriter('/content/clusters_dataset.xlsx') as writer:
            self.healthy_data.to_excel(writer, sheet_name='healthy', index=False)
            self.aphasia_data.to_excel(writer, sheet_name='aphasia', index=False)
