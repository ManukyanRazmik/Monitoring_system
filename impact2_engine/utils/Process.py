import datetime as dt
from typing import Optional, Any
# from numpy.typing import ArrayLike
from warnings import warn
import pandas as pd
import numpy as np


class ProcessSafety:
    """Preprocessing routines.
    """

    def __init__(self,
                 data_path: str,  # csv(','), utf-8, 1st line header
                 contents: dict[str, Any],
                 na_filter: bool = True):

        self.data_path = data_path
        self.contents = contents
        self.na_filter = na_filter
        self.data = pd.DataFrame()
        self.missing: Optional[pd.DataFrame] = None
        self.update_data()


    def update_data(self) -> None:
        """Check, load and preprocess data at request"""

        self.__check_contents()
        self.__load_data(na_filter = self.na_filter)

        if self.na_filter:
            self.__handle_na()

        self.__discretize_demo()


    def __check_contents(self) -> None:

        try:
            with open(self.data_path, 'r', encoding = 'utf-8') as csv:
                header = csv.readline().rstrip().split(',')
        except ValueError:
            print("""Ensure path is correct and contains 'utf-8' csv-file,
                  with ',' delimeter and header on the 1st line.""")

        names = [col['name'] for cols in self.contents.values() for col in cols]
        absent = [name for name in names if name not in header]
        duplicate = [col for col in header if header.count(col) > 1]

        if len(absent) > 0:
            raise KeyError(
                f"Columns {', '.join(absent)} are required but absent."
            )

        if len(duplicate) > 0:
            raise KeyError(
                f"Columns {', '.join(duplicate)} are not unique."
            )

        for col in self.contents['SEV']:
            col['aes'] = [event for event in col['aes'] if event in header]


    def __load_data(self, na_filter: bool = True) -> None:

        key_names = {key: [col['name'] for col in cols]
                     for key, cols in self.contents.items()}
        col_names = [name for names in key_names.values() for name in names]
        found_aes = [col['aes'] for col in self.contents['SEV']
                     if col['var'] == 'all_ae'][0]
        bool_flags = key_names['POP'] + key_names['SEV'] + found_aes
        col_types = (
            dict.fromkeys(key_names['CAT'], 'category') |
            dict.fromkeys(key_names['DEM'], 'float') |
            dict.fromkeys(key_names['IDS'] + bool_flags, 'object')
        )

        self.data = pd.read_csv(
            filepath_or_buffer = self.data_path,
            usecols = col_names + found_aes,
            dtype = col_types,
            parse_dates = key_names['DAT'],
            na_filter = na_filter  # detect missing value markers if True
        )

        self.data.replace(
            to_replace = dict.fromkeys(
                bool_flags,
                {'Yes': True, 'No': False, np.nan: False}
            ),
            inplace = True
        )

        # replace columns by standardized variable names ?

        name_to_var = {col['name']: col['var']
                              for cols in self.contents.values()
                              for col in cols}
        self.data.columns = [name_to_var.get(item, item)
                             for item in self.data.columns]

        dates = self.contents['DAT'][0]['var']
        self.data.set_index(dates, inplace = True)
        self.data.sort_index(inplace = True)
        # self.data.reset_index(inplace = True)


    def __handle_na(self) -> None:

        if self.data.isnull().values.any():
            warn(
                """Found missing values. Keep only compelete cases.
                Inspect 'missing' field to find the source of NA.
                Disable na_filter to read the original data."""
            )
            self.missing = self.data[self.data.isna().any(axis = 1)]
            self.data.dropna(axis = 0, how = 'any', inplace = True)


    def __discretize_demo(self) -> None:

        for col in self.contents['DEM']:

            if (isinstance(col['bin'], list) and
                isinstance(col['lvl'], list) and
                len(col['bin']) == len(col['lvl']) + 1 and
                len(col['lvl']) > 1):

                self.data[col['var']] = pd.cut(
                    self.data[col['var']],
                    bins = col['bin'],
                    labels = col['lvl'],
                    right = False
                )

            elif (isinstance(col['bin'], int) and
                  isinstance(col['lvl'], list) and
                  len(col['lvl']) == col['bin']):

                self.data[col['var']] = pd.qcut(
                    self.data[col['var']],
                    q = col['bin'],
                    labels = col['lvl']
                )

            else:
                raise ValueError(
                    """Numerical columns in contents of 'DEM' should contain:
                    the 'lvl' list of labels and 'bin' into dicrete categories,
                    provided either as cutpoints (list) or number of bins (int)."""
                )


    def save_data(self,
                  path: str) -> None:
        """Write processed data to csv file.

        Args:
            path (str): Path to file.
        """

        self.data.reset_index().to_csv(
            path_or_buf = path,
            index = False
        )


    def save_missing(self,
                     path: str) -> None:
        """Write missing data to csv file.

        Args:
            path (str): Path to file.
        """

        if self.missing is None:
            raise KeyError(
                "No missing data was found"
            )


        self.missing.reset_index().to_csv(
            path_or_buf = path,
            index = False
        )



class ProcessPlasma:
    """Plasma Collection monitoring module
    """

    def __init__(self,
                 data_path: str,  # csv(','), utf-8, 1st line header
                 contents: dict[str, Any],
                 na_filter: bool = True):

        self.data_path = data_path
        self.contents = contents
        self.na_filter = na_filter
        self.missing: Optional[pd.DataFrame] = None
        self.data = pd.DataFrame()
        self.update_data()


    def update_data(self) -> None:
        """Check, load and preprocess data at request"""

        self.__check_contents()
        self.__load_data(na_filter = self.na_filter)

        if self.na_filter:
            self.__handle_na()

        self.__arrange()
        self.__discretize_demo()
        self.__preprocess_data()


    def __check_contents(self) -> None:

        try:
            with open(self.data_path, 'r', encoding = 'utf-8') as csv:
                header = csv.readline().rstrip().split(',')
        except ValueError:
            print("""Ensure path is correct and contains 'utf-8' csv-file,
                  with ',' delimeter and header on the 1st line.""")

        names = [col['name'] for cols in self.contents.values()
                 for col in cols]
        absent = [name for name in names if name not in header]
        duplicate = [col for col in header if header.count(col) > 1]

        if len(absent) > 0:
            raise KeyError(
                f"Columns {', '.join(absent)} are required but absent."
            )

        if len(duplicate) > 0:
            raise KeyError(
                f"Columns {', '.join(duplicate)} are not unique."
            )

        for col in self.contents['SEV']:
            col['aes'] = [event for event in col['aes'] if event in header]


    def __load_data(self, na_filter: bool = True) -> None:

        key_names = {key: [col['name'] for col in cols]
                     for key, cols in self.contents.items()}
        col_names = [name for names in key_names.values() for name in names]
        bool_flags = key_names['POP'] + key_names['SEV']
        col_types = (
            dict.fromkeys(key_names['CAT'], 'category') |
            dict.fromkeys(key_names['DEM'] + key_names['VOL'], 'float') |
            dict.fromkeys(key_names['IDS'] + key_names['DTS'] + bool_flags, 'object')
        )

        self.data = pd.read_csv(
            filepath_or_buffer = self.data_path,
            usecols = col_names,
            dtype = col_types,
            na_values = {col['name']: col['na'] for col in self.contents['VOL']},
            na_filter = na_filter  # detect missing value markers if True
        )

        self.data.replace(  # convert to bool
            to_replace = dict.fromkeys(
                bool_flags,
                {'Yes': True, 'No': False, np.nan: False}
            ),
            inplace=True
        )

        for col in self.contents['DTS']:  # faster than parse_dates
            self.data[col['name']] = pd.to_datetime(
                self.data[col['name']], format = col['format']
            )

        name_to_var = {col['name']: col['var']
                       for cols in self.contents.values()
                       for col in cols}
        self.data.columns = [name_to_var.get(item, item)
                             for item in self.data.columns]


    def __handle_na(self) -> None:

        if self.data.isnull().values.any():
            warn(
                """Found missing values. Keep only compelete cases.
                Inspect 'missing' field to find the source of NA.
                Disable na_filter to read the original data."""
            )
            self.missing = self.data[self.data.isna().any(axis = 1)]
            self.data.dropna(axis = 0, how = 'any', inplace = True)


    def __arrange(self,
                  var: Optional[str] = None,
                  reset: bool = False) -> None:

        if var is None:
            var = self.contents['DTS'][0]['var']  # provided no NA

        self.data.set_index(var, inplace = True)
        self.data.sort_index(inplace = True)

        if reset:
            self.data.reset_index(inplace = True)


    def __discretize_demo(self) -> None:

        for col in self.contents['DEM']:

            if (isinstance(col['bin'], list) and
                isinstance(col['lvl'], list) and
                len(col['bin']) == len(col['lvl']) + 1 and
                    len(col['lvl']) > 1):

                self.data[col['var']] = pd.cut(
                    self.data[col['var']],
                    bins=col['bin'],
                    labels=col['lvl'],
                    right=False
                )

            elif (isinstance(col['bin'], int) and
                  isinstance(col['lvl'], list) and
                  len(col['lvl']) == col['bin']):

                self.data[col['var']] = pd.qcut(
                    self.data[col['var']],
                    q=col['bin'],
                    labels=col['lvl']
                )

            else:
                raise ValueError(
                    """Numerical columns in contents of 'DEM' should contain:
                    the 'lbl' list of labels and 'bin' into dicrete categories,
                    provided either as cutpoints (list) or number of bins (int)."""
                )

    def __preprocess_data(self) -> None:
        """Enrich data by derived variables"""

        self.data['duration'] = self.data['proc_end'] - self.data['proc_start']

        if np.any(self.data['duration'] <= dt.timedelta(0)):
            raise ValueError(
                "Make sure 'proc_end' > 'proc_start'"
            )

        self.data['duration_minutes'] = self.data['duration'].dt.total_seconds() / 60
        self.data['speed'] = self.data['actual_vol'] / self.data['duration_minutes']
        self.data['yield'] = self.data['actual_vol'] / self.data['target_vol']
        self.data['yield_resid'] = 1 - self.data['yield']
        self.data['week'] = pd.Series(
            self.data.index
        ).dt.to_period('W').dt.start_time.values

        del self.data['duration']

    def save_data(self,
                  path: str) -> None:
        """Write processed data to csv file.

        Args:
            path (str): Path to file.
        """

        self.data.reset_index().to_csv(
            path_or_buf = path,
            index = False
        )


    def save_missing(self,
                     path: str) -> None:
        """Write missing data to csv file.

        Args:
            path (str): Path to file.
        """

        if self.missing is None:
            raise KeyError(
                "No missing data was found"
            )

        self.missing.reset_index().to_csv(
            path_or_buf = path,
            index = False
        )


class ProcessProfile:
    """Plasma Collection monitoring module
    """

    def __init__(self,
                 data_path: str,  # csv(','), utf-8, 1st line header
                 contents: dict[str, Any],
                 na_filter: bool = True):

        self.data_path = data_path
        self.contents = contents
        self.na_filter = na_filter
        self.missing: Optional[pd.DataFrame] = None
        self.data = pd.DataFrame()
        self.update_data()


    def update_data(self) -> None:
        """Check, load and preprocess data at request"""

        self.__check_contents()
        self.__load_data(na_filter = self.na_filter)

        if self.na_filter:
            self.__handle_na()

        self.__arrange()
        self.__discretize_demo()
        self.__preprocess_data()


    def __check_contents(self) -> None:

        try:
            with open(self.data_path, 'r', encoding = 'utf-8') as csv:
                header = csv.readline().rstrip().split(',')
        except ValueError:
            print("""Ensure path is correct and contains 'utf-8' csv-file,
                  with ',' delimeter and header on the 1st line.""")

        names = [col['name'] for cols in self.contents.values()
                 for col in cols]
        absent = [name for name in names if name not in header]
        duplicate = [col for col in header if header.count(col) > 1]

        if len(absent) > 0:
            raise KeyError(
                f"Columns {', '.join(absent)} are required but absent."
            )

        if len(duplicate) > 0:
            raise KeyError(
                f"Columns {', '.join(duplicate)} are not unique."
            )

        for col in self.contents['SEV']:
            col['aes'] = [event for event in col['aes'] if event in header]


    def __load_data(self, na_filter: bool = True) -> None:

        key_names = {key: [col['name'] for col in cols]
                     for key, cols in self.contents.items()}
        col_names = [name for names in key_names.values() for name in names]
        found_aes = [col['aes'] for col in self.contents['SEV']
                     if col['var'] == 'all_ae'][0]
        bool_flags = key_names['POP'] + key_names['SEV'] + found_aes
        col_types = (
            dict.fromkeys(key_names['CAT'], 'category') |
            dict.fromkeys(key_names['DEM'] + key_names['VOL'], 'float') |
            dict.fromkeys(key_names['IDS'] + key_names['DTS'] + bool_flags, 'object')
        )

        self.data = pd.read_csv(
            filepath_or_buffer = self.data_path,
            usecols = col_names + found_aes,
            dtype = col_types,
            na_values = {col['name']: col['na'] for col in self.contents['VOL']},
            na_filter = na_filter  # detect missing value markers if True
        )

        self.data.replace(  # convert to bool
            to_replace = dict.fromkeys(
                bool_flags,
                {'Yes': True, 'No': False, np.nan: False}
            ),
            inplace = True
        )

        for col in self.contents['DTS']:  # faster than parse_dates
            self.data[col['name']] = pd.to_datetime(
                self.data[col['name']], format = col['format']
            )

        name_to_var = {col['name']: col['var']
                       for cols in self.contents.values()
                       for col in cols}
        self.data.columns = [name_to_var.get(item, item)
                             for item in self.data.columns]


    def __handle_na(self) -> None:

        if self.data.isnull().values.any():
            warn(
                """Found missing values. Keep only compelete cases.
                Inspect 'missing' field to find the source of NA.
                Disable na_filter to read the original data."""
            )
            self.missing = self.data[self.data.isna().any(axis = 1)]
            self.data.dropna(axis = 0, how = 'any', inplace = True)


    def __arrange(self,
                  var: Optional[str] = None,
                  reset: bool = False) -> None:

        if var is None:
            var = self.contents['DTS'][0]['var']  # provided no NA

        self.data.set_index(var, inplace = True)
        self.data.sort_index(inplace = True)

        if reset:
            self.data.reset_index(inplace = True)


    def __discretize_demo(self) -> None:

        for col in self.contents['DEM']:

            if (isinstance(col['bin'], list) and
                isinstance(col['lvl'], list) and
                len(col['bin']) == len(col['lvl']) + 1 and
                    len(col['lvl']) > 1):

                self.data[col['var'] + '_cat'] = pd.cut(
                    self.data[col['var']],
                    bins = col['bin'],
                    labels = col['lvl'],
                    right = False
                )

            elif (isinstance(col['bin'], int) and
                  isinstance(col['lvl'], list) and
                  len(col['lvl']) == col['bin']):

                self.data[col['var'] + '_cat'] = pd.qcut(
                    self.data[col['var']],
                    q = col['bin'],
                    labels = col['lvl']
                )

            else:
                raise ValueError(
                    """Numerical columns in contents of 'DEM' should contain:
                    the 'lbl' list of labels and 'bin' into dicrete categories,
                    provided either as cutpoints (list) or number of bins (int)."""
                )

    def __preprocess_data(self) -> None:
        """Enrich data by derived variables"""

        self.data['duration'] = self.data['proc_end'] - self.data['proc_start']

        if np.any(self.data['duration'] <= dt.timedelta(0)):
            raise ValueError(
                "Make sure 'proc_end' > 'proc_start'"
            )

        self.data['duration_minutes'] = self.data['duration'].dt.total_seconds() / 60
        self.data['speed'] = self.data['actual_vol'] / self.data['duration_minutes']
        self.data['yield'] = self.data['actual_vol'] / self.data['target_vol']
        self.data['yield_resid'] = 1 - self.data['yield']

        del self.data['duration']

    def save_data(self,
                  path: str) -> None:
        """Write processed data to csv file.

        Args:
            path (str): Path to file.
        """

        self.data.reset_index().to_csv(
            path_or_buf = path,
            index = False
        )


    def save_missing(self,
                     path: str) -> None:
        """Write missing data to csv file.

        Args:
            path (str): Path to file.
        """

        if self.missing is None:
            raise KeyError(
                "No missing data was found"
            )

        self.missing.reset_index().to_csv(
            path_or_buf = path,
            index = False
        )
