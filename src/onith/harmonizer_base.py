from dataclasses import dataclass
from IPython.display import HTML, display
import pandas as pd
import os

@dataclass 
class HarmonizerBase:
    temp_dir: str
    output_dir: str
    project_name: str
    sample_column: str = "USUBJID"
    study_id_column: str = "STUDYID"
    group_id_column: str = "ARMCD"
    
    def harmonize_group_ids_by_number(self,
        df,
        study_id_column: str = "STUDYID",
        group_id_column: str = "ARMCD",
        include_recovery_animals: bool = False
    ):
        """
        Adjusts group ID values for each study ID group so that the minimum group ID becomes 0.

        Background:
            Treatment groups (whether the animal was in the control group or in one of the treatment groups)
            of each sample (typically animals) are usually indicated with integers, the lowest number being
            assigned to the control group. However, there is no standardization in whether to start with 0 or 1
            for the control group. This function ensures consistency by normalizing the group IDs to start at 0.

        Parameters:
            df (pd.DataFrame): A DataFrame containing at least the study ID and group ID columns.
            study_id_column (str): The name of the column representing the study identifier. Default is 'STUDYID'.
            group_id_column (str): The name of the column representing the group identifier (e.g., treatment group). Default is 'ARMCD'.
            include_recovery_animals (bool): If True, keeps recovery animals by treating them as a separate study group.
                                            If False, removes rows with 'R' in the group ID.

        Returns:
            pd.DataFrame: A DataFrame with adjusted group ID values where the minimum group ID per study is 0.
        """
        df = df.copy()

        # Backup original group ID
        df[f"original_{group_id_column}"] = df[group_id_column]

        # Identify recovery rows
        is_recovery = df[group_id_column].astype(str).str.contains("R")

        if include_recovery_animals:
            # Tag recovery animals with a modified STUDYID
            df.loc[is_recovery, study_id_column] = df.loc[is_recovery, study_id_column].astype(str) + "_REC"
            # Remove 'R' and convert to numeric
            df[group_id_column] = df[group_id_column].astype(str).str.replace("R", "", regex=False)
        else:
            # Exclude recovery animals
            df = df[~is_recovery]

        # Convert to numeric
        df[group_id_column] = pd.to_numeric(df[group_id_column], errors='coerce')

        # Normalize group IDs per study
        for study_id in df[study_id_column].unique():
            while True: # repeatedly substract 1 from the group id until the lowest group id is 0
                min_group_id = df.loc[df[study_id_column] == study_id, group_id_column].min()
                if pd.isna(min_group_id) or min_group_id == 0:
                    break
                df.loc[df[study_id_column] == study_id, group_id_column] -= min_group_id

        return df


    def harmonize_group_ids_by_name(self, 
        df,
        group_name_column: str = "ARM",
        group_id_column: str = "ARMCD",
        control_keywords: list = None
    ):
        """
        Scans group names for control-related keywords and assigns 0 to the group ID if a match is found.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            group_name_column (str): Column name containing group descriptions.
            group_id_column (str): Column name for group IDs to be updated.
            control_keywords (list): List of keywords indicating control groups. Default includes common terms.

        Returns:
            pd.DataFrame: Updated DataFrame with group IDs set to 0 where control keywords are found.
        """
        df = df.copy()

        if control_keywords is None:
            control_keywords = ["control", "vehicle", "placebo", "ctrl", "water"]

        control_keywords = [kw.lower() for kw in control_keywords]

        # Ensure all values are strings and handle NaNs safely
        mask = df[group_name_column].astype(str).str.lower().apply(
            lambda name: any(kw in name for kw in control_keywords)
        )

        df.loc[mask, group_id_column] = 0

        return df


    def extract_metadata(self, 
                        df, 
                        output_dir:str, 
                        project_name:str, 
                        sample_column:str = "USUBJID", 
                        study_id_column:str = "STUDYID", 
                        group_id_column:str = "ARMCD", 
                        group_name_column:str = "ARM", 
                        include_recovery_animals:bool = True, 
                        control_keywords = None):
        """
        Extracts and harmonizes metadata from a study dataset and saves it as a CSV file.

        This function reduces the dataset to key metadata columns, 
        harmonizes treatment group IDs (both numerically and by keyword-based group name matching), 
        and saves the resulting metadata to the specified output directory.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing study data.
            output_dir (str): Directory path where the metadata CSV will be saved.
            project_name (str): Name of the project, used in the output filename.
            sample_column (str): Column name identifying individual animals or samples.
            study_id_column (str): Column name for study identifiers. Default is 'STUDYID'.
            group_id_column (str): Column name for group IDs. Default is 'ARMCD'.
            group_name_column (str): Column name for group descriptions. Default is 'ARM'.
            control_keywords (list): Optional list of keywords to identify control groups by name.

        Returns:
            pd.DataFrame: The harmonized metadata DataFrame.
        """
        
        # reduce columns
        df = df[[sample_column, study_id_column, group_id_column, group_name_column]]
        
        # harmonize group ids
        display(HTML("<p>Harmonizing the treatment group ids by assigned integer and group name keywords... </p>"))
        df = self.harmonize_group_ids_by_number(df, study_id_column=study_id_column, group_id_column=group_id_column, include_recovery_animals=include_recovery_animals)
        df = self.harmonize_group_ids_by_name(df, group_name_column=group_name_column, group_id_column=group_id_column, control_keywords=control_keywords)
        
        # save metadata in output_dir
        metadata_path = os.path.join(output_dir, f"metadata_{project_name}.csv")
        df.to_csv(metadata_path)
        display(HTML(f"<p>The resulting metadata dataframe was saved to {metadata_path}.</p>"))
        display(HTML(f"""<h3>Please review the metadata file to ensure that all control animals have been correctly assigned the group ID '0'. This is essential for accurate z-score calculations during the subsequent domain-specific harmonization.<br>
                    Also remove all animals from the list that you don't want to include into the harmonization process. Whatever animal is not present in the metadata file, will not be included in the domain-specific harmonization process.</h3>"""))
        
        return df


    def filter_by_metadata(self, df, output_dir, project_name, sample_column, metadata_path=None):
        """
        Filters the input DataFrame to include only rows with sample IDs present in the metadata file.

        Parameters:
            df (pd.DataFrame): The input DataFrame to be filtered.
            metadata_path (str): Optional path to the metadata CSV file. If None, a default path is constructed.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        if metadata_path is None:
            metadata_path = os.path.join(output_dir, f"metadata_{project_name}.csv")

        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            if sample_column in metadata.columns:
                df = df[df[sample_column].isin(metadata[sample_column])]
                print(f"df shape after filtering by metadata ({metadata_path}): ", df.shape)
            else:
                print(f"Warning: Column '{sample_column}' not found in metadata.")
        else:
            print(f"Warning: Metadata file not found at {metadata_path}. No filtering applied.")

        return df