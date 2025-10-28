from IPython.display import display, HTML
import pandas as pd
import numpy as np
from .ontology_utils import *
from .mi_harmonizer import *
from .lb_harmonizer import *
from dataclasses import dataclass
from collections import defaultdict
from collections import Counter


@dataclass
class BWHarmonizer(LBHarmonizer):
    sample_column: str = "USUBJID"
    measurement_column: str = "BWTEST"
    unit_column: str = "BWSTRESU"
    value_column: str = "BWSTRESC"
    day_column: str = "BWDY"
    domain = "bw"
    
    
    def clean_bw(self, df: pd.DataFrame, metadata_path: str = None) -> pd.DataFrame:
        """
        Filters and prepares body weight data for analysis.
        """
        
        print("dataframe shape before filtering by metadata: ", df.shape)
        df = df.copy()
        df = df[[self.study_id_column, self.sample_column, self.measurement_column, self.value_column, self.unit_column, self.day_column]]
        df = self.filter_by_metadata(df, output_dir=self.output_dir, project_name=self.project_name, sample_column=self.sample_column, metadata_path=metadata_path)

        print("df shape before removing na: ", df.shape)
        df = df.dropna()
        print("df shape after removing na: ", df.shape)
        
        return df
    
    
    def explore_parameter_frequency(self, df: pd.DataFrame) -> None:
        """
        Displays frequency of parameter combinations per sample.
        """
        
        # Create a mapping from each parameter type to the set of samples
        bwtest_map = df.groupby(self.measurement_column)[self.sample_column].apply(set).to_dict()

        # Create a dictionary to track which parameter types each sample has
        usubjid_to_types = defaultdict(set)
        for test, usubjids in bwtest_map.items():
            for usubjid in usubjids:
                usubjid_to_types[usubjid].add(test)

        # Count how many samples fall into each unique combination of parameteres
        combination_counts = Counter(frozenset(tests) for tests in usubjid_to_types.values())

        # Print result
        print(f"Number of samples ({self.sample_column}) per parameter type ({self.measurement_column}):")
        for combo, count in combination_counts.items():
            print(f"{', '.join(combo) if combo else 'None'}: {count}")
            
        return 

        
    def filter_for_terminal_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters body weight data to retain only terminal measurements.
        """
        
        # Check for NaN values in the day column
        if df[self.day_column].isna().any():
            raise ValueError(f"Column '{self.day_column}' contains NaN values. Please clean the data before proceeding.")

        # Filter to keep only the one row with the maximum day per sample
        filtered_df = df.loc[df.groupby(self.sample_column)[self.day_column].idxmax()]
        print("DataFrame shape after reducing to only one weight per animal (the terminal weight):", filtered_df.shape)
        
        filtered_df[self.measurement_column] = "Terminal Body Weight"

        return filtered_df

    
    def pivot_bw(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots terminal body weight z-scores by sample.
        """

        om_pivot = pd.pivot_table(df[[self.sample_column, "z-score"]], index=self.sample_column, values="z-score", fill_value=np.nan)

        return om_pivot