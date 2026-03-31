from IPython.display import display, HTML
import pandas as pd
import numpy as np
from .ontology_utils import *
from .mi_harmonizer import *
from .lb_harmonizer import *
from dataclasses import dataclass
from .harmonizer_base import *
from .export_logger import *
import seaborn as sns
import matplotlib.pyplot as plt



@dataclass
class OMHarmonizer(LBHarmonizer, MIHarmonizer):
    sample_column: str = "USUBJID"
    organ_column: str = "OMSPEC"
    measurement_column: str = "OMTEST"
    unit_column: str = "OMSTRESU"
    value_column: str = "OMSTRESC"
    domain = "om"
    

    def clean_om(self, df: pd.DataFrame, metadata_path: str = None) -> pd.DataFrame:
        """
        Cleans and filters organ measurements for analysis.
        """
        
        print("df shape after import: ", df.shape)
        df = df[[self.study_id_column, self.sample_column, self.organ_column, self.measurement_column, self.unit_column, self.value_column]]
        df = df.copy()
        df[f"original {self.organ_column}"] = df[self.organ_column].copy()
        df = harmonize_formatting(df, [self.organ_column])
        df = self.filter_by_metadata(df, output_dir=self.output_dir, project_name=self.project_name, sample_column=self.sample_column, metadata_path=metadata_path)

        return df
    
    
    def apply_mapping_om(self, df: pd.DataFrame, organ_mapping: pd.DataFrame) -> pd.DataFrame:
        """
        Applies organ mapping to harmonize organ names and IDs.
        """
        
        df_organ_dict_name = organ_mapping.set_index(f"original {self.organ_column}")["mapped_organ_main_term_name"].to_dict()
        df_organ_dict_id = organ_mapping.set_index(f"original {self.organ_column}")["mapped_organ_main_term_ID"].to_dict()

        df["harmonized_organ"] = df[f"original {self.organ_column}"].map(df_organ_dict_name)
        df["harmonized_organ_id"] = df[f"original {self.organ_column}"].map(df_organ_dict_id)
        
        df = df.drop(columns=f"original {self.organ_column}")
        df = df.drop_duplicates()
        
        return df
    
    
    def explore_parameter_frequency(self, df: pd.DataFrame) -> None:
        """
        Displays a heatmap of measurement frequency per organ and parameter.
        """
        
        # Calculate statistics
        stats = df.groupby([self.measurement_column, 'harmonized_organ'])[self.sample_column].nunique().reset_index()
        stats = stats.rename(columns={self.sample_column: 'unique_USUBJID_count'})

        # Pivot for heatmap
        heatmap_data = stats.pivot(index="harmonized_organ", columns=self.measurement_column, values="unique_USUBJID_count").fillna(0)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", cbar=True)
        plt.title("Number of measurements per sample, organ and parameter type")
        plt.xlabel("Parameter Type")
        plt.ylabel("Organ")
        plt.tight_layout()
        plt.show()
        
        display(HTML("<p>To-Do: Decide on a parameter type and configure the 'parameter_type' parameter of the following function respectively.</p>"))
        
        return
    

    def prepare_om_for_control_stats(self, df: pd.DataFrame, parameter_type: str) -> pd.DataFrame:
        """
        Filters organ measurements to a specific parameter type for analysis.
        """
        
        parameter_type = "Weight"
        
        print(f"dataframe shape before reducing to parameters_type '{parameter_type}': ", df.shape)
        df = df[df[self.measurement_column] == parameter_type]
        print(f"dataframe shape after reducing to parameters_type '{parameter_type}': ", df.shape)
        
        df = df.drop(columns=self.measurement_column)
        self.measurement_column = "harmonized_organ"
        
        return df
                
    
    def pivot_om(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots organ measurement z-scores by sample and organ.
        """

        # Generate pivot table
        om_pivot = pd.pivot_table(df[[self.sample_column, "harmonized_organ", "z-score"]], index=self.sample_column, columns="harmonized_organ", values="z-score", fill_value=np.nan)

        return om_pivot