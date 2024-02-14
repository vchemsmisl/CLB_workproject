import pandas as pd


class DataExtraction:
    def __init__(self, link: str) -> None:
        self.dataset_norm = pd.read_excel(link, sheet_name='healthy')
        self.dataset_aphasia = pd.read_excel(link, sheet_name='aphasia')
        self.category_types = {'animals': 'a',
                               'professions': 'b',
                               'cities': 'c'}

    def get_ids(self, sheet_name: str = 'healthy') -> int:
        """
        Getting ID column
        """
        if sheet_name == 'healthy':
            return self.dataset_norm['ID']
        return self.dataset_aphasia['ID']

    def get_column_name(self, category: str, lexemes: str):
        """
        Get column name from category and type of included lexemes;
        lexemes: clean | clean-all-lexemes
        """
        return f'C6({self.category_types.get(category)})-{lexemes}'

    def get_series(self, sheet_name: str, category: str, lexemes: str = 'clean'):
        """
        Getting one of 12 columns:
          from one of the 2 pages of the dataset
          from one of the 3 categories
          from one of the types of lexemes

        sheet_name: healthy | aphasia
        category: animals | professions | cities
        lexemes: clean | clean-all-lexemes
        """
        if sheet_name == 'healthy':
            return self.dataset_norm[self.get_column_name(category, lexemes)]

        return self.dataset_aphasia[self.get_column_name(category, lexemes)]