import pandas as pd
from src.data_extraction import DataExtraction


class ClustersData:
    def __init__(self, extractor: DataExtraction) -> None:
        self.extractor = extractor
        self.id_healthy = extractor.get_ids()
        self.id_aphasia = extractor.get_ids('aphasia')
        self.healthy_data = pd.DataFrame(self.id_healthy)
        self.aphasia_data = pd.DataFrame(self.id_aphasia)

    def get_df(self, sheet):
        """
        Getting a df depending on the sheet name
        :param sheet: sheet name, healthy | aphasia
        :return: pd.DataFrame
        """
        if sheet == 'healthy':
            return self.healthy_data
        return self.aphasia_data

    @staticmethod
    def get_column_name(category: str, lexemes: str) -> str:
        category_types = {'animals': 'a',
                          'professions': 'b',
                          'cities': 'c'}
        return f'C6({category_types.get(category)})-{lexemes}'

    @staticmethod
    def avg_length(row: pd.Series) -> float:
        """
        Count average length of cell values in a row
        """
        return sum(len(x) for x in row) / len(row)

    def add_column(self, sheet_name: str, category: str,
                   lexemes: str, clusters: pd.Series) -> None:
        """
        Adding a column with clusters
        of a specified sheet, category and type of lexemes fullness
        """
        if sheet_name == 'healthy':
            column_name = self.get_column_name(category, lexemes)
            self.healthy_data[column_name] = clusters

        else:
            column_name = self.get_column_name(category, lexemes)
            self.aphasia_data[column_name] = clusters

    def count_switches(self, sheet_name: str,
                       category: str, lexemes: str) -> None:
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
        Count mean number of clusters for each row
        """
        clean_columns = [self.get_column_name(category, 'clean') for category in ['animals',
                                                                                  'professions',
                                                                                  'cities']]
        all_columns = [self.get_column_name(category, 'clean-all-lexemes') for category in ['animals',
                                                                                            'professions',
                                                                                            'cities']]

        if sheet_name == 'healthy':
            self.healthy_data.insert(loc=7,
                                     column='Average_cluster_number_clean',
                                     value=self.healthy_data[clean_columns].apply(self.avg_length, axis=1))
            self.healthy_data.insert(loc=14,
                                     column='Average_cluster_number_all',
                                     value=self.healthy_data[all_columns].apply(self.avg_length, axis=1))
        else:
            self.aphasia_data.insert(loc=7,
                                     column='Average_cluster_number_clean',
                                     value=self.aphasia_data[clean_columns].apply(self.avg_length, axis=1))
            self.aphasia_data.insert(loc=14,
                                     column='Average_cluster_number_all',
                                     value=self.aphasia_data[all_columns].apply(self.avg_length, axis=1))

    def save_excel(self) -> None:
        """
        Saving data with clusters to an Excel file
        """
        with pd.ExcelWriter('/content/clusters_dataset.xlsx') as writer:
            self.healthy_data.to_excel(writer, sheet_name='healthy', index=False)
            self.aphasia_data.to_excel(writer, sheet_name='aphasia', index=False)
