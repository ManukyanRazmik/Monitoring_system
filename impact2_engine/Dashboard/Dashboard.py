"""Dashboard module for IMPACT study"""

# %% Import libraries

__all__ = ['Dashboard']

import datetime as dt
from typing import Optional, Union, Any
from warnings import warn
import numpy as np
import pandas as pd
import sidetable


# %% Dashboard class definition and methods


class Dashboard:
	"""
	Dashboard module
	"""

	def __init__(self,
				 data_path: str,  # csv(','), utf-8, 1st line header
				 contents: dict[str, Any],
				 na_filter: bool = True):

		self.data_path = data_path
		self.contents = contents
		self.na_filter = na_filter
		self.missing: Optional[pd.DataFrame] = None
		self.update_data()


	def update_data(self) -> None:
		"""
		Check, load and preprocess data at request

		"""
		self.__load_data()
		self.__check_contents()
		if self.na_filter:
			self.__handle_na()

		self.__group = [col['var'] for col in self.contents['GRP']
						 if col['name'] == 'GROUP'][0]

		self.__donor = {col['name'] : col['var']  for col in self.contents['DON']}

		

	def __load_data(self) -> None:        
		
		try:
			self.data = pd.read_parquet(path = self.data_path)
		except ValueError:
			print("""Ensure path is correct and contains parquet file""")
		except ImportError:
			print("""Ensure either 'pyarrow' or 'fastparquet' is installed""")
		
		 

	def __check_contents(self) -> None:
		

		header = self.data.columns.to_list()

		names = [col['var'] for cols in self.contents.values() for col in cols
							if col['name'] not in ['OVERDRAWS', 'UNDERDRAWS']]
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


	   
	def __handle_na(self) -> None:

		if self.data.isnull().values.any():
			warn(
				"""Found missing values. Keep only compelete cases.
				Inspect 'missing' field to find the source of NA.
				Disable na_filter to read the original data."""
			)
			self.missing = self.data[self.data.isna().any(axis = 1)]
			self.data.dropna(axis = 0, how = 'any', inplace = True)



	def summary_donation(self) -> pd.DataFrame:
		"""
		Calculates the statistics of donors and donations

		"""
		graph1_1 = pd.pivot_table(
								  data = self.data, 
								  index = self.__group,
								  values = [self.__donor['DONOR_ID'], self.__donor['WITHDRAW']],
								  aggfunc={
										self.__donor['DONOR_ID']: [pd.Series.nunique, 'count'],
										self.__donor['WITHDRAW'] : 'sum'
										   }
								  )

		graph1_1.columns = ['n_donors', 'n_donations', 'n_withdrawals']
		graph1_1 = graph1_1.stb.subtotal().T
		graph1_1.loc['repeat_don_rate', :] = ( graph1_1.iloc[0,:] / graph1_1.iloc[1,:])
		graph1_1.columns = ['group_A', 'group_B', 'total']

		return graph1_1


	def summary_demographic(self, by:str, donor_level:bool = True):
		"""
		Donation statistics by groups
		
		"""
		cat_dict = {col['name'] : col['var'] for col in self.contents['CAT']}

	   
		if by == 'SITE':
			graph2_1 = self.data.groupby([cat_dict[by]])[self.__donor['DONOR_ID']]			
			if donor_level:
				graph2_1 = graph2_1.nunique()
			else:
				graph2_1 = graph2_1.count()
		
		elif by in ['GENDER', 'STATUS']:
			graph2_1 = self.data.groupby([self.__group, cat_dict[by]])[self.__donor['DONOR_ID']]
			tot_group = self.data.groupby([self.__group])[self.__donor['DONOR_ID']]

			if donor_level:
				graph2_1 = graph2_1.nunique() / tot_group.nunique()
			else:
				graph2_1 = graph2_1.count() / tot_group.count()
			
			graph2_1.index = list(map(lambda x: x[0] + '_' + x[1].lower(), graph2_1.index))

		elif by in ['AGE', 'BMI', 'WEIGHT', 'HEMO']:
			
			graph2_1 = pd.pivot_table(data=self.data,
									  index=self.__group, 
									  columns=cat_dict[by], 
									  values=self.__donor['DONOR_ID'], 
									  aggfunc='nunique' if donor_level else 'count'
									  )
			
			graph2_1 = graph2_1.stb.subtotal().T  
			graph2_1[['A', 'B']] = graph2_1[['A', 'B']].div(graph2_1['grand_total'], axis=0)
			graph2_1.index = map(lambda x: x.lower(), graph2_1.index)
			graph2_1.fillna(0, inplace = True)
		else:
			raise ValueError(
				f"Possible grouping values are {', '.join(cat_dict.keys())}."
				)

		return graph2_1


	def summary_metrc(self):

		metrics = {col['name'] : col['var'] for col in self.contents['PLZ']}
		metric_cols = list(metrics.values())

		under_lvl = [col['lvl'] for col in self.contents['PLZ'] if col['name'] == 'UNDERDRAWS'][0]
		self.data[metrics['OVERDRAWS']] = (self.data[metrics['TARGET']] > self.data[metrics['ACTUAL']]).astype(int)
		self.data[metrics['UNDERDRAWS']] = (self.data[metrics['TARGET']] < self.data[metrics['ACTUAL']]).astype(int)

		for i in under_lvl:    
			under_col_percent = f'{metrics["UNDERDRAWS"]}_{i}'
			metric_cols.append(under_col_percent)
	
			self.data[under_col_percent] =(i * self.data[metrics['ACTUAL']] > self.data[metrics['TARGET']]).astype(int)


		graph3_1 = pd.pivot_table(data=self.data, 
								  index=self.__group, 
								  values=metric_cols + [self.__donor['DONOR_ID']],  
								  aggfunc= dict.fromkeys(metric_cols, sum) | {self.__donor['DONOR_ID']: 'count'})

		graph3_1 = graph3_1.stb.subtotal()

		graph3_1[['average_actual', 'average_target', 'average_yieald']] = \
				 graph3_1[[metrics['ACTUAL'], metrics['TARGET'], metrics['YIELD']]].div(graph3_1[self.__donor['DONOR_ID']], axis = 0)

		graph3_1 = graph3_1.drop(columns=[self.__donor['DONOR_ID'], metrics['YIELD']], axis = 1).T

		return graph3_1