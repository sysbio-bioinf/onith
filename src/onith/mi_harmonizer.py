from IPython.display import display, HTML
from rapidfuzz import fuzz, process
import pandas as pd
import numpy as np
from .ontology_utils import *
from typing import Literal
import ast
import json
from dataclasses import dataclass
from .harmonizer_base import *
from .export_logger import *
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import os 
from pathlib import Path


@dataclass
class MIHarmonizer(HarmonizerBase):
    sample_column: str = "USUBJID"
    organ_column: str = "MISPEC"
    lesion_column: str = "MIORRES"
    domain = "mi"

    def clean_mi(self, df, metadata_path = None):
        """
        Cleans and harmonizes the formatting of the input dataframe containing the histopathological finding descriptions. 
        Reduces the dataset to necessary column (animal, organ and lesion) and the samples that also are present in the metadata file.
        
        Parameters:
        df (pd.DataFrame): The input dataframe containing the histopathological findings structured as: sample column (here: animal ids), organ column, lesion column (description of the histopathological alteration).
        """

        # replace NaN in lesion column (MISTRESC or MIORRES) with "NORMAL"
        df[self.lesion_column] = df[self.lesion_column].fillna("NORMAL")
        
        # replace NaN in self.organ_column by "whole body" (so that it will not be removed later, when reducing to organ panel)
        df[self.organ_column] = df[self.organ_column].fillna("WHOLE BODY")
        
        # copy the original columns before harmonizing their formatting
        df[f"original {self.organ_column}"] = df[self.organ_column].copy()
        df[f"original {self.lesion_column}"] = df[self.lesion_column].copy()
        
        # harmonize formatting of terms
        print("df shape after import: ", df.shape)
        df = harmonize_formatting(df, [self.organ_column])
        df = harmonize_formatting(df, [self.lesion_column])
        
        # reduce columns 
        df = df[[self.sample_column, f"original {self.organ_column}", self.organ_column, f"original {self.lesion_column}", self.lesion_column]]
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        print("df shape after removing duplicates, harmonizing terms and reducing to relevant columns: ", df.shape)
        
        # load metadata and filter
        df = self.filter_by_metadata(df, output_dir=self.output_dir, project_name=self.project_name, sample_column=self.sample_column, metadata_path=metadata_path)

        return df


    def prepare_for_automatic_mapping(self, df, input_type : Literal["organ","lesion"]):
        """
        Prepares a DataFrame for automatic mapping by cleaning and filtering based on the input type (either organ or lesion).

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing data to be prepared for mapping.
            - If input_type == "lesion", the function removes animal ID columns and entries labeled as "NORMAL" in the lesion column to make the mapping process clearer.
            - If input_type == "organ", the function removes the animal ID column and lesion column.

        Duplicate rows are removed after column filtering.
        """
        print("df shape before preparing for automatic mapping: ", df.shape)

        if input_type == "lesion":
            # remove animal id columns and NORMAL lesion entries from input dataset before mapping
            df = df.drop(columns = self.sample_column).drop_duplicates()
            print("df shape after removing USUBJIDs and drop duplicates: ",  df.shape)
            df = remove_normal_mi_entries(df, self.lesion_column)
            print("df shape after removing NORMAL entries: ", df.shape)

        elif input_type == "organ":
            # Define the columns you want to drop
            df = df.copy()
            df = df[[self.organ_column, f"original {self.organ_column}"]].drop_duplicates()
         
        return df

    
    def remove_grade_info(self, df):
        """
        Additional and optional harmonization step to facilitate the mapping of lesion terms to the hpath ontology. 
        Removes information about the grade of the lesion from the lesion term and drops duplicates afterwards.
        """
        # remove organ and grade information from lesion terms
        df[self.lesion_column] = df.apply(
            lambda row: (
                row[self.lesion_column].replace(row[self.organ_column], "", 1).strip()
                if row[self.organ_column] in row[self.lesion_column]
                else row[self.lesion_column]
            ).lstrip(",").lstrip("S,").strip(),
            axis=1
        )
        df[self.lesion_column] = df[self.lesion_column].str.replace(r"\d+","", regex=True)
        df[self.lesion_column] = df[self.lesion_column].str.replace("GRADE ","",regex=False)
        df[self.lesion_column] = df[self.lesion_column].str.replace("MINIMAL","",regex=False)
        df[self.lesion_column] = df[self.lesion_column].str.replace("MILD","",regex=False)
        df[self.lesion_column] = df[self.lesion_column].str.replace("SEVERE","",regex=False)
        df[self.lesion_column] = df[self.lesion_column].str.replace("SLIGHT","",regex=False)
        df[self.lesion_column] = df[self.lesion_column].str.replace("MODERATE","",regex=False)
        
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        
        return df


    def fuzzy_mapping(self, 
                    df, 
                    terms, 
                    input_type : Literal["lesion", "organ"],
                    include_list: bool, 
                    threshold: int = 50, 
                    auto_accept_threshold: int = 92):
        """
        Performs fuzzy string matching to map lesion or organ terms in a DataFrame to the respective reference ontology.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing the terms to be mapped.
        terms : list or pd.Series
            The list of reference terms (coming from the ontology) to match against.
        input_type : Literal["lesion", "organ"]
            Specifies whether to map lesion or organ terms. Determines which column in the DataFrame is used.
        include_list : bool
            If True, includes all matches above the threshold as a list when the best match is below the auto_accept_threshold.
            If False, no match is assigned unless the best score exceeds the auto_accept_threshold.
        threshold : int, default=50
            Minimum fuzzy match score required to consider a term as a potential match.
        auto_accept_threshold : int, default=92
            Score above which the best match is automatically accepted without listing alternatives.

        Returns:
        --------
        pd.DataFrame
            The input DataFrame with additional columns:
            - For "lesion": "lesion_mapped_term" and "lesion_fuzzy_score"
            - For "organ": "organ_mapped_term" and "organ_fuzzy_score"

        Notes:
        ------
        - Uses `fuzz.token_sort_ratio` from the `rapidfuzz` library for scoring.
        - Assumes `self.lesion_column` and `self.organ_column` are defined and point to the correct columns in `df`.
        """
        mapped_terms = []
        mapped_scores = []
        
        if input_type == "lesion":
            column = self.lesion_column
        elif input_type == "organ":
            column = self.organ_column

        for term in df[column]:
            matches = process.extract(term, terms, scorer=fuzz.token_sort_ratio, limit=None)
            filtered_matches = [(match, score) for match, score, _ in matches if score >= threshold]

            if filtered_matches:
                # Sort matches by score, descending
                filtered_matches.sort(key=lambda x: x[1], reverse=True)
                best_match, best_score = filtered_matches[0]

                if best_score >= auto_accept_threshold:
                    mapped_terms.append(best_match)
                    mapped_scores.append(best_score)
                elif include_list:
                    mapped_terms.append([match for match, _ in filtered_matches])
                    mapped_scores.append([score for _, score in filtered_matches])
                else:
                    mapped_terms.append(None)
                    mapped_scores.append(None)
            else:
                mapped_terms.append(None)
                mapped_scores.append(None)

        if input_type == "lesion":
            df["lesion_mapped_term"] = mapped_terms
            df["lesion_fuzzy_score"] = mapped_scores
        elif input_type == "organ":
            df["organ_mapped_term"] = mapped_terms
            df["organ_fuzzy_score"] = mapped_scores

        return df


    def fuzzy_mapping_unmapped(self,
                    df, 
                    input_type : Literal["lesion","organ"],
                    include_list: bool, 
                    terminology,
                    term_type: Literal["main", "synonym"]):
        """
        Applies fuzzy mapping to unmapped lesion or organ terms in the input DataFrame using either the ontology's main terms or their respective synonyms.
        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing lesion or organ terms to be mapped.
        input_type : Literal["lesion", "organ"]
            Specifies whether to map lesion or organ terms and therefore. Determines which columns are used and updated.
        include_list : bool
            If True, includes all matches above the fuzzy threshold as a list when the best match is below the auto-accept threshold.
            If False, only the best match is used if it meets the auto-accept threshold.
        terminology : dict
            Determines whether the organ or lesion ontology is used for the mapping. Must include keys:
            - "name": list or Series of main terms
            - "synonym": list or Series of synonym terms
        term_type : Literal["main", "synonym"]
            Specifies whether to use main terms or synonyms for fuzzy matching.
            
        Returns:
        --------
        pd.DataFrame
            The updated DataFrame with fuzzy mapping results added for previously unmapped terms. Adds or updates:
            - "lesion_mapped_term" / "organ_mapped_term"
            - "lesion_mapping_type" / "organ_mapping_type"
            
        Notes:
        ------
        - Only terms with no existing mapped term are processed.
        - Mapping type labels indicate whether the match was to a main term, synonym, or requires manual review.
        - Relies on fuzzy mapping using the rapidfuzz library
        """

        if input_type == "lesion":
            mapped_term_column = "lesion_mapped_term"
            mapping_type_column = "lesion_mapping_type"
        elif input_type == "organ":
            mapped_term_column = "organ_mapped_term"
            mapping_type_column = "organ_mapping_type"
            
        # only search mappings for terms that don't have a good match yet
        mask_unmapped = df[mapped_term_column].isna()
        df_unmapped = df[mask_unmapped].copy()

        if term_type == "main":
            df_unmapped = self.fuzzy_mapping(df=df_unmapped, input_type=input_type, include_list=include_list, terms=terminology["name"])
            df_unmapped[mapping_type_column] = "fuzzy_mapping_to_main_term"
        elif term_type == "synonym":
            df_unmapped = self.fuzzy_mapping(df=df_unmapped, input_type=input_type, include_list=include_list, terms=terminology["synonym"])
            if include_list:
                df_unmapped[mapping_type_column] = "manual_mapping_needed"
            else:
                df_unmapped[mapping_type_column] = "fuzzy_mapping_to_synonym"
        else:
            raise ValueError("term_type must be either 'main' or 'synonym'")

        df.update(df_unmapped)        
    
        return df


    def automatic_mapping(self, df, input_type : Literal["organ", "lesion"]) -> pd.DataFrame:
        """
        Automatically maps lesion or organ terms that occur in the input dataset to reference ontologies using a multi-step strategy that includes both direct and fuzzy matching.
        Fuzzy matching relies on word similarity and accepts mappings above a configurable similarity threshold. The similarity is typically measured using metrics such as Levenshtein distance.

        The mapping process includes:
        1. Direct mapping to main terms (100% match).
        2. Direct mapping to synonym terms (100% match).
        3. Direct mapping to synonyms after removing grade information (lesion only).
        4. Iterative fuzzy mapping to main terms and synonyms with decreasing specificity.
        5. Final fuzzy mapping with candidate lists for manual review.

        Parameters:
        -----------
        df : pd.DataFrame
            The input dataset containing lesion or organ terms to be mapped.
        input_type : Literal["organ", "lesion"]
            Specifies whether to map organ or lesion terms. Determines which columns and ontology are used.

        Returns:
        --------
        pd.DataFrame
            The input DataFrame with additional columns for mapped terms, mapping type, and fuzzy match score:
            - For lesions: "lesion_mapped_term", "lesion_mapping_type", "lesion_fuzzy_score"
            - For organs: "organ_mapped_term", "organ_mapping_type", "organ_fuzzy_score"

        Notes:
        ------
        - Displays progress and summary statistics using HTML output.
        - Saves intermediate results to an csv file in `self.temp_dir`.
        - Unmapped terms after all mapping steps are flagged for manual review.
        - Uses helper methods: `prepare_for_automatic_mapping`, `remove_grade_info`, `fuzzy_mapping_unmapped`.
        """

        if input_type == "lesion":
            column = self.lesion_column
            df = self.prepare_for_automatic_mapping(df, "lesion")
            mapped_term_column = "lesion_mapped_term"
            mapping_type_column = "lesion_mapping_type"
            fuzzy_score_column = "lesion_fuzzy_score"
        elif input_type == "organ":
            column = self.organ_column
            df = self.prepare_for_automatic_mapping(df, "organ")
            mapped_term_column = "organ_mapped_term"
            mapping_type_column = "organ_mapping_type"
            fuzzy_score_column = "organ_fuzzy_score"
        
        dict = load_and_prepare_dict(input_type)
        
        # Create a new column for mapping results, initialized with None
        df[mapped_term_column] = None
        df[mapping_type_column] = None
        df[fuzzy_score_column] = None
        
        # Level 1: Direct mapping to main terms
        mask_level1 = df[column].isin(dict["name"]) # what rows have a direct mapping to a dict main term
        df.loc[mask_level1, [mapped_term_column]] = df.loc[mask_level1, column] 
        df.loc[mask_level1, mapping_type_column] = "direct_mapping_to_main_term"
        df.loc[mask_level1, fuzzy_score_column] = 100.0 

        # Level 2: Direct mapping to synonyms (for unmapped terms)
        mask_level2 = df[column].isin(dict["synonym"]) & df[mapped_term_column].isna() # which remaining rows have a direct mapping to a dict synonym
        df.loc[mask_level2, mapped_term_column] = df.loc[mask_level2, column] 
        df.loc[mask_level2, mapping_type_column] = "direct_mapping_to_synonym"
        df.loc[mask_level2, fuzzy_score_column] = 100.0 
        
        # Level 3: Direct mapping to synonyms after removing grade information (only for lesion)
        if input_type == "lesion":
            df = self.remove_grade_info(df)
            mask_level3 = df[column].isin(dict["synonym"]) & df[mapped_term_column].isna() # what of the remaining rows have a direct mapping to a dict synonym
            df.loc[mask_level3, mapped_term_column] = df.loc[mask_level3, column] 
            df.loc[mask_level3, mapping_type_column] = "direct_mapping_to_synonym"
            df.loc[mask_level3, fuzzy_score_column] = 100.0 
        
        # print feedback of how successful direct mapping was
        mapping_percentage = df[~df[mapped_term_column].isna()].shape[0] / df.shape[0] * 100
        
        display(HTML(f"<p><strong>{round(mapping_percentage)}% of entries were successfully mapped to the ontology with a 100% match</strong></p>"))
        
        if mapping_percentage < 100:
            display(HTML("<p>Applying fuzzy mapping for remaining entries now (this can take a few minutes)...</p>"))
            
            # Level 3–8: Iterative fuzzy mapping with progressive term simplification
            for level in range(3, 9):
                # Fuzzy match against main terms
                df = self.fuzzy_mapping_unmapped(df, input_type=input_type, include_list=False, terminology=dict, term_type="main")
                
                # Fuzzy match against synonym terms
                df = self.fuzzy_mapping_unmapped(df, input_type=input_type, include_list=False, terminology=dict, term_type="synonym")
                
                if level < 7:
                    # Prepare for next level by stripping the last segment
                    df[column] = df[column].apply(strip_last_segment)
            
            # Last Round: Fuzzy match against synonym terms and store lists of mapping candidates (fuzzy matches with at least 70% score)
            df = self.fuzzy_mapping_unmapped(df, input_type=input_type, include_list=True, terminology=dict, term_type="synonym")

            # Count how many lesions were matched with at least 95% similarity 
            # Print summary results of fuzzy mapping 

            if input_type == "organ":
                remaining_terms = df[~
                    df[mapped_term_column].apply(lambda x: isinstance(x, str))
                ][self.organ_column].drop_duplicates().shape[0]
                remaining_rows = df[~
                    df[mapped_term_column].apply(lambda x: isinstance(x, str))
                ].shape[0]
                mapping_percentage = round(((df.shape[0] - remaining_rows) / df.shape[0]) * 100)
                
            elif input_type == "lesion":
                remaining_terms = df[~
                    df[mapped_term_column].apply(lambda x: isinstance(x, str))
                ][self.lesion_column].drop_duplicates().shape[0]
                remaining_rows = df[~
                    df[mapped_term_column].apply(lambda x: isinstance(x, str))
                ].shape[0]
                mapping_percentage = round(((df.shape[0] - remaining_rows) / df.shape[0]) * 100)                
            
            display(HTML(f"<p><strong>{mapping_percentage}% of entries were matched to the dict ontology with at least 95% similarity after fuzzy mapping</strong></p>"))
            display(HTML(f"there are {remaining_terms} remaining terms that have to be mapped manually. -> To do so, launch the manual mapping editor in the next step..."))

            # save intermediate results
            temp_file = os.path.join(self.temp_dir, f"intermediate_results_automatic_{self.domain}_{input_type}_mapping.csv")
            df.to_csv(temp_file) 
            
            display(HTML(f"<p>The temporary results are saved to: {temp_file}</p>"))

        display(HTML("<p><strong>These are the (temporary) results of the automatic mapping:</strong></p>"))
        display(df)
            
        return df


    def prepare_for_manual_mapping(self, df, column : str, input_type : Literal["lesion","organ"]) -> pd.DataFrame:
        """
        Prepares a DataFrame for manual mapping by filtering rows where fuzzy mapping was performed but no match with a score above the configurable threshold was reached.
        Also reduces the DataFrame to only the relevant columns needed for manual review and completion in the sharable streamlit application.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing mapping results, including candidate lists (derived from fuzzy matching).
        column : str
            The name of the column containing the mapping candidates as list.
        input_type : Literal["lesion", "organ"]
            Specifies whether the mapping is for lesion or organ terms. Determines which columns are retained.

        Returns:
        --------
        pd.DataFrame
            A filtered and simplified DataFrame containing only rows with list-type mapping candidates and the necessary columns
            for manual mapping:
            - For "lesion": [organ_column, original lesion column, mapped term column]
            - For "organ": [original organ column, mapped term column]

        Notes:
        ------
        - Handles cases where table may load lists as strings by evaluating them safely.
        - Assumes `self.organ_column` and `self.lesion_column` are defined.
        """

        # Helper: Detect lists ot NA (means that no good match was found -> instead list of mapping candidates was stored as table entry, based on fuzzy score)
        def is_list(val):
            if val == "" or val is None: 
                return True
            if isinstance(val, list):
                return True
            if isinstance(val, str):
                try:
                    v = eval(val)
                    return isinstance(v, list)
                except:
                    return False
            return False

        # only keep rows with list
        df = df[df[column].apply(is_list)].reset_index(drop=True)
        
        # reduce to necessary columns for manual completion
        if input_type == "lesion":
            df = df[[self.organ_column,f"original {self.lesion_column}",column]]
        elif input_type == "organ":
            df = df[[f"original {self.organ_column}",column]]
        
        return df
    

    def compile_custom_dropdown_lists(self, df, input_type = Literal["lesion", "organ"]) -> pd.DataFrame:
        """
        Enriches the mapped term column (containing the best fuzzy matched synonym and main terms from the previous step) in the DataFrame 
        by ensuring each list also contains all unique main terms from the reference ontology. 
        These custom list per term will then be used as basis for the dropdown menu in the manual mapping editor application.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame where the mapped term column contains lists of candidate terms.
        input_type : Literal["lesion", "organ"]
            Specifies whether to operate on lesion or organ terms. Determines which column is updated.

        Returns:
        --------
        pd.DataFrame:
            The updated DataFrame where each list in the mapped term column includes all unique main terms 
            from the ontology, with original list items first and missing terms appended.

        Notes:
        ------
        - Uses `load_and_prepare_dict(input_type)` to retrieve the reference ontology.
        - Only modifies rows where the mapped term column contains a list.
        - Ensures no duplicates and maintains the original order of existing terms.
        """

        def process_list(lst):
            if lst is None or lst == 'NA':
                lst = []
            seen = set()
            deduped_lst = [x for x in lst if not (x in seen or seen.add(x))]
            to_append = [term for term in main_terms if term not in deduped_lst]
            return deduped_lst + to_append

        lesion_dict = load_and_prepare_dict(input_type)
        main_terms = lesion_dict["name"].drop_duplicates().tolist()
        
        if input_type == "lesion":
            mapped_term_column = "lesion_mapped_term"
        elif input_type == "organ":
            mapped_term_column = "organ_mapped_term"     
        
        # Apply the function to each row, only if the value is a list
        df[mapped_term_column] = df[mapped_term_column].apply(
            lambda x: process_list(x) if isinstance(x, list) or x is None else x)

        return df


    def jsons_for_streamlit(self, df, input_type : Literal["organ", "lesion"]):
        """
        Converts the dataFrame containing the mapping candidates into a list of dictionaries formatted for Streamlit display and saves them as a JSON file for use in the manual mapping editor.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing original terms and mapping candidates (as lists or strings).
        input_type : Literal["organ", "lesion"]
            Specifies whether the mapping is for lesion or organ terms. Determines which columns are used.

        Returns:
        --------
        str
            The file path to the saved JSON file.

        Notes:
        ------
        - Converts string representations of lists into actual Python lists using `ast.literal_eval`.
        - Filters and formats each row into a dictionary suitable for display in the Streamlit-based manual mapping tool.
        - Displays HTML messages to inform the user about the saved file and the next steps.
        """
        
        if input_type == "lesion":
            mapped_term_column = "lesion_mapped_term"
        elif input_type == "organ":
            mapped_term_column = "organ_mapped_term"
        
        manual_mappings = []
        for _, row in df.iterrows():
            # string to list if needed
            row_term_order = row[mapped_term_column]
            
            if isinstance(row_term_order, str):
                try:
                    row_term_order = ast.literal_eval(row_term_order)
                except (ValueError, SyntaxError):
                    row_term_order = []

            if input_type == "lesion":
                manual_mappings.append({
                    "Organ": row[self.organ_column],
                    "Original Term": row[f"original {self.lesion_column}"],
                    "Row Term Order": row_term_order
                })
            elif input_type == "organ":
                manual_mappings.append({
                    "Original Term": row[f"original {self.organ_column}"],
                    "Row Term Order": row_term_order
                })                
                
        # Store remaining terms for manual mapping as json for streamlit in self.temp_dir
        json_path_mapping = os.path.join(self.temp_dir, f"{self.project_name}_{self.domain}_{input_type}_terms_for_manual_mapping.json")
        with open(json_path_mapping, "w") as f:
            json.dump(manual_mappings, f)
            
        display(HTML(f"<p>The input for the manual mapping editor was saved to: {json_path_mapping}</p>"))
        display(HTML(f"<p>The editor will now open (as new tab in your internet explorer, can take a moment)...</p>"))
        
        return json_path_mapping


    def launch_streamlit(self, sharable, json_path_mapping, input_type, json_path_dict, show_term_info):
        """
        Launches the Streamlit-based manual mapping editor as a subprocess, passing required arguments.

        Parameters:
        -----------
        sharable : str
            Flag indicating whether the session should be launched in a sharable mode (as empty application without the data already loaded, json files with application content to be shared separately).
        json_path_mapping : str
            Path to the JSON file containing mapping candidates for manual review.
        input_type : str
            Specifies whether the mapping is for "lesion" or "organ". Used to name the progress file.
        json_path_dict : str
            Path to the JSON file that delivers the content for the manual mappind editor.
        show_term_info : str
            Flag indicating whether to display an info card with additional term information in the editor.

        Side Effects:
        -------------
        - Starts the Streamlit app (`manual_mapping_editor.py`) as a separate process.
        - Passes the mapping file (application content), dictionary file (term info for infor card), progress file path (to save progress from the application), and flags as command-line arguments.
        - Configures the default path for saving progress as: `{project_name}_{self.domain}_{input_type}_progress_manual_mapping.json` in `self.temp_dir`. Can be adjusted in the application.
        """
        module_dir = os.path.dirname(__file__)  
        app_path = os.path.join(module_dir, "manual_mapping_editor.py")
        save_dir = os.path.join(self.temp_dir, f"{self.project_name}_{self.domain}_{input_type}_progress_manual_mapping.json")
        subprocess.Popen(["streamlit", "run", app_path, json_path_mapping, json_path_dict, save_dir, sharable, show_term_info])
        
        return
        

    def launch_manual_mapping(self, df, input_type : Literal["organ", "lesion"], sharable : str = "False") -> None:
        """
        Prepares and triggers to launch the streamlit-based manual mapping editor. 

        This method:
        1. Filters and formats the input DataFrame for manual mapping.
        2. Compiles enriched dropdown lists for candidate terms.
        3. Generates JSON files for Streamlit input.
        4. Launches the Streamlit app with the appropriate configuration.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing mapping candidates.
        input_type : Literal["organ", "lesion"]
            Specifies whether the mapping is for lesion or organ terms.
        sharable : str, default="False"
            Flag indicating whether the Streamlit session should be launched in sharable mode.

        Notes:
        ------
        - For lesion mapping, additional term information is included via `hpath_term_info_to_json`.
        - The Streamlit app (`manual_mapping_editor.py`) must be available and accessible.
        """
        
        if input_type == "lesion":
            mapped_term_column = "lesion_mapped_term"
            show_term_info : str = "True"
            
            # Store current state of the ontology as json for the term info card in streamlit
            lesion_dict_info = hpath_term_info_to_json(input_type)
            json_path_dict = os.path.join(self.temp_dir, f"{self.project_name}_{self.domain}_{input_type}_dict_term_info.json")
            with open(json_path_dict, "w") as f:
                json.dump(lesion_dict_info, f)
            
        elif input_type == "organ":
            mapped_term_column = "organ_mapped_term"  
            show_term_info : str = "False"      
            json_path_dict = ""
            original_col = f"original {self.organ_column}"
            
            # Create the original orgaan column if it doesn't exist
            if original_col not in df.columns:
                df[original_col] = df[self.organ_column].copy()
                
        df = self.prepare_for_manual_mapping(df, mapped_term_column, input_type)
        df = self.compile_custom_dropdown_lists(df, input_type)
        json_path_mapping = self.jsons_for_streamlit(df, input_type)
        self.launch_streamlit(sharable, json_path_mapping, input_type, json_path_dict, show_term_info)
        
        return


    def add_new_synonyms_to_obo(self, obo_file_path, df, input_type:Literal["organ", "lesion"]):
        """
        Adds new manually reviewed synonyms (as exported from the manual mapping editor and found by the fuzzy matching) to the respective local OBO ontology file.

        This method:
        - Parses the OBO file while preserving its structure.
        - Identifies terms in the ontology by their IDs.
        - Filters the input DataFrame to exclude exact matches (100% score).
        - Adds new synonyms to the appropriate term blocks in the OBO file.
        - Updates the header with a timestamp indicating the last manual completion.
        - Adds term specific timestamps for version control

        Parameters:
        -----------
        obo_file_path : str
            Path to the existing OBO file to be updated.
        df : pd.DataFrame
            DataFrame containing original terms, fuzzy scores, and mapped ontology term IDs.
        input_type : Literal["organ", "lesion"]
            Specifies whether the synonyms are for organ or lesion terms. Determines which columns are used.

        Returns:
        --------
        List[str]
            A list of strings representing the updated OBO file content, ready to be written back to the locally stored OBO file.

        Notes:
        ------
        - Only synonyms with a fuzzy score less than 100 are considered for addition.
        - Only synonyms that are not already listed as synonym are considered.
        - A timestamp is added or updated in the OBO file header to indicate when manual synonyms were last added.
        - Assumes the DataFrame contains the following columns:
            - `mapped_{input_type}_main_term_ID`
            - `original {self.lesion_column}` or `original {self.organ_column}`
            - `{input_type}_fuzzy_score`
        """
        # Read OBO file while preserving structure
        with open(obo_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Identify the header section
        header_end_index = next(i for i, line in enumerate(lines) if line.strip().startswith("[Term]"))

        # Generate manual completion timestamp
        current_date = datetime.now().strftime("%Y-%m-%d")
        manual_completion_entry = f"last manual completion with additional synonyms: {current_date}\n"

        header_lines = lines[:header_end_index]
        if any("last manual completion with additional synonyms:" in line for line in header_lines):
            header_lines = [manual_completion_entry if "last manual completion with additional synonyms:" in line else line for line in header_lines]
        else:
            header_lines.append(manual_completion_entry)

        # Parse [Term] blocks
        terms = []
        current_term = []
        for line in lines[header_end_index:]:
            if line.strip().startswith("[Term]"):
                if current_term:
                    terms.append(current_term)
                current_term = [line]
            else:
                current_term.append(line)
        if current_term:
            terms.append(current_term)
        
        # Build a mapping from id to term block index
        id_to_index = {}
        for idx, term in enumerate(terms):
            for line in term:
                if line.strip().startswith("id:"):
                    term_id = line.strip().split("id:")[1].strip()
                    id_to_index[term_id] = idx
                    break

        # Filter new synonyms based on fuzzy mapping score
        if input_type == "lesion":
            fuzzy_score_column = "lesion_fuzzy_score"
            column = self.lesion_column
        elif input_type == "organ":
            fuzzy_score_column = "organ_fuzzy_score"
            column = self.organ_column
            
        df_filtered = pd.to_numeric(df[fuzzy_score_column], errors="coerce").fillna(0)
        df_filtered = df[~(df[fuzzy_score_column] == 100)]

        # Add synonyms to the correct term blocks, but only if the synonym doesnt already exist
        for _, row in df_filtered.iterrows():
            term_id = row[f"mapped_{input_type}_main_term_ID"]
            original_synonym = row[f"original {column}"]
            new_synonym = f'synonym: "{original_synonym}" MANUAL [{current_date}]\n'
            idx = id_to_index.get(term_id)

            if idx is not None:
                term = terms[idx]
                existing_synonyms = []
                for l in term:
                    if l.strip().startswith("synonym:"):
                        match = re.search(r'synonym:\s*"([^"]+)"', l)
                        if match:
                            existing_synonyms.append(match.group(1).strip())

                if original_synonym in existing_synonyms:
                    continue  # Skip if synonym already exists

                # Find all synonym lines and is_a line
                synonym_indices = [i for i, l in enumerate(term) if l.strip().startswith("synonym:")]
                is_a_indices = [i for i, l in enumerate(term) if l.strip().startswith("is_a:")]

                # Insert after last synonym, or before is_a, or at end
                if synonym_indices:
                    insert_at = synonym_indices[-1] + 1
                    term.insert(insert_at, new_synonym)
                elif is_a_indices:
                    insert_at = is_a_indices[0]
                    term.insert(insert_at, new_synonym)
                else:
                    blank_indices = [i for i, l in enumerate(term) if l.strip() == ""]
                    insert_at = blank_indices[0] if blank_indices else len(term)
                    term.insert(insert_at, new_synonym)

        # Reconstruct the updated file
        updated_dict = header_lines + ["\n"] + [line for term in terms for line in term]

        return updated_dict


    def unite_and_save_mappings(self, 
                                df,
                                update_dict : bool,
                                integrate_manual_mapping : bool,
                                input_type : Literal["organ","lesion"],
                                json_path_mapping = None,
                                json_path_progress = None
                                ) -> pd.DataFrame:
        """
        This method:
        - Loads manual mapping metadata and user selections (from the manual mapping editor) from JSON files.
        - Merges manual mappings with previously auto-mapped terms.
        - Optionally updates the ontology (OBO file) with new synonyms.
        - Saves the final mapping results.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing automatically mapped terms.
        update_dict : bool
            If True, updates the local ontology file with newly mapped synonyms.
        integrate_manual_mapping : bool
            If True, it loads all term metadata and term selections (by importing the "..._terms_for_manual_mapping.json" and "..._progress_manual_mapping.json" from the temp_dir) from the manual mapping editor and integrates it into the automatic mappings from the step before.
            If False, it doesn't search for the two json files from the manual mapping editor, but instead only uses the results of the automatic mapping as basis. 
        input_type : Literal["organ", "lesion"]
            Specifies whether the mapping is for organ or lesion terms.
        json_path_mapping : str, optional
            Path to the JSON file containing manual mapping metadata (input data for the manual mapping editor). If not provided, a default path is used based on specified temp_dir and project_name.
        json_path_progress : str, optional
            Path to the JSON file containing user-selected mappings. If not provided, a default path is used based on specified temp_dir and project_name.

        Returns:
        --------
        pd.DataFrame
            The final combined mapping DataFrame, including both automatic and manual mappings.

        Notes:
        ------
        - Manual mappings are labeled with `manual_mapping` in the mapping type column.
        - Fuzzy scores for manual mappings are set to None.
        - The ontology is updated only if `update_dict` is True, and the updated version is saved with a backup.
        - Final results are saved as an csv file in `self.output_dir`.
        - Assumes the presence of helper methods: `fetch_dict_main_terms`, `add_new_synonyms_to_obo`, and `save_updated_dict`.
        """
        
        if input_type == "lesion":
            mapped_term_column = "lesion_mapped_term"
            mapping_type_column = "lesion_mapping_type"
            fuzzy_score_column = "lesion_fuzzy_score"
            column = self.lesion_column
        elif input_type == "organ":
            mapped_term_column = "organ_mapped_term"
            mapping_type_column = "organ_mapping_type"
            fuzzy_score_column = "organ_fuzzy_score"
            column = self.organ_column
        
        if integrate_manual_mapping:
            
            # set paths for metadata (input for streamlit) and selections (as selected from the dropdown menu by the user and then saved as progress) 
            if json_path_mapping is None:
                json_path_mapping = os.path.join(self.temp_dir, f"{self.project_name}_{self.domain}_{input_type}_terms_for_manual_mapping.json")
            if json_path_progress is None:
                json_path_progress = os.path.join(self.temp_dir, f"{self.project_name}_{self.domain}_{input_type}_progress_manual_mapping.json")
                    
            # load metadata of manually mapped terms
            if not os.path.exists(json_path_mapping):
                raise FileNotFoundError(f"JSON file not found: {json_path_mapping}")
            
            with open(json_path_mapping, "r") as f:
                mapping_man = json.load(f)
            
            mapping_man = pd.DataFrame(mapping_man)
            mapping_man = mapping_man.drop(columns="Row Term Order")
            
            if input_type == "lesion":
                mapping_man = mapping_man.rename(columns={"Organ": self.organ_column, "Original Term": f"original {self.lesion_column}"})
            elif input_type == "organ":
                mapping_man = mapping_man.rename(columns={"Original Term": f"original {self.organ_column}"})
            
            # load term selections from the manual mapping editor
            if not os.path.exists(json_path_progress):
                raise FileNotFoundError(f"JSON file not found: {json_path_progress}")

            with open(json_path_progress, "r") as f:
                term_selection = json.load(f)
                
            term_selection = term_selection.get("selections")

            # merge term selections with metadata of manually mapped terms                
            mapping_man[mapped_term_column] = term_selection
            mapping_man[column] = mapping_man[f"original {column}"]
            mapping_man = harmonize_formatting(mapping_man, [column])
            mapping_man[mapping_type_column] = "manual_mapping"
            mapping_man[fuzzy_score_column] = None
        
            # merge manually with automatically mapped terms
            mapping_auto = df[df[mapped_term_column].apply(lambda x: isinstance(x, str))]
            mapping = pd.concat([mapping_auto, mapping_man], axis=0, ignore_index=True)
        
        #mapping = df[df[mapped_term_column].apply(lambda x: isinstance(x, str))]
        
        else:
            mapping = df.copy()
            mapping[mapped_term_column] = mapping[column]
            mapping[mapping_type_column] = "no_mapping_needed"
            mapping[fuzzy_score_column] = None
        
        # fetch main term information
        mapping = self.fetch_dict_main_terms(mapping, input_type)
        
        # update lesion_dict with new synonyms if wished
        if update_dict == True:
            # add mapped original terms to as dict syononyms
            
            if input_type == "lesion":
                base_path = Path(__file__).resolve().parent / "dict"
                obo_file_path = os.path.join(base_path,"hpath_ontology.obo")
                
            elif input_type == "organ":
                base_path = Path(__file__).resolve().parent / "dict"
                obo_file_path = os.path.join(base_path,"organ_ontology.obo")
                
            updated_dict = self.add_new_synonyms_to_obo(obo_file_path, mapping, input_type)
            
            # convert back to obo and save
            subfolder = save_updated_dict(obo_file_path, updated_dict, input_type)
            
            display(HTML("<p>Your local copy of the ontology was updated with the mapped terms from your dataset and saved to: dict </p>"))
            display(HTML(f"<p>The previous local version of the ontology can be restored from: {subfolder}</p>"))
        
        # save mapping results
        mapping_file = os.path.join(self.output_dir, f"{self.project_name}_final_{self.domain}_{input_type}_mapping.csv")
        mapping.to_csv(mapping_file) 

        display(HTML("<p><strong>These are the final results of the mapping:</strong></p>"))
        display(mapping)
        display(HTML(f"<p>The mapping results are also saved to: {mapping_file}</p>"))
        
        return mapping


    def fetch_dict_main_terms(self, df, input_type : Literal["organ", "lesion"]):
        dict = load_and_prepare_dict(input_type)
        dict_synonym_to_name = dict.set_index("synonym")["name"].to_dict()
        dict_name_to_id = dict.set_index("name")["id"].to_dict()
        
        df = df.copy()
        mask1 = df[f"{input_type}_mapped_term"].notna()
        df.loc[mask1, f"mapped_{input_type}_main_term_name"] = df.loc[mask1, f"{input_type}_mapped_term"].map(dict_synonym_to_name)

        mask2 = df[f"mapped_{input_type}_main_term_name"].notna()
        df.loc[mask2, f"mapped_{input_type}_main_term_ID"] = df.loc[mask2, f"mapped_{input_type}_main_term_name"].map(dict_name_to_id)

        return df

        
    def apply_mapping_mi(self, df, organ_mapping, lesion_mapping):
        """
        Applies the previously united organ and lesion mappings and their corresponding ontology IDs to the input dataset.

        Parameters:
        -----------
        df : pd.DataFrame
            The original dataset containing raw organ and lesion terms.
        organ_mapping : pd.DataFrame
            A DataFrame containing mappings from original organ terms to standardized ontology terms and IDs.
            Must include columns: "original MISPEC", "mapped_organ_main_term_name", "mapped_organ_main_term_ID".
        lesion_mapping : pd.DataFrame
            A DataFrame containing mappings from original lesion terms to standardized ontology terms and IDs.
            Must include columns: f"original {self.lesion_column}", "mapped_lesion_main_term_name", "mapped_lesion_main_term_ID".

        Returns:
        --------
        pd.DataFrame
            The input DataFrame with additional columns:
            - "harmonized_organ": standardized organ term (default: "WHOLE BODY" if no match)
            - "harmonized_organ_id": corresponding ontology ID
            - "harmonized_lesion": standardized lesion term (default: "NORMAL" if no match)
            - "harmonized_lesion_id": corresponding ontology ID

        Notes:
        ------
        - Displays summary statistics comparing the number of unmapped terms before and after harmonization.
        - Assumes `self.lesion_column` and `self.organ_column` are defined and point to the original term columns.
        - Uses `.map()` to apply mappings and `.fillna()` to assign default values for unmatched terms.
        """
        
        df_organ_dict_name = organ_mapping.set_index(f"original {self.organ_column}")["mapped_organ_main_term_name"].to_dict()
        df_organ_dict_id = organ_mapping.set_index(f"original {self.organ_column}")["mapped_organ_main_term_ID"].to_dict()
        df_lesion_dict_name = lesion_mapping.set_index(f"original {self.lesion_column}")["mapped_lesion_main_term_name"].to_dict()
        df_lesion_dict_id = lesion_mapping.set_index(f"original {self.lesion_column}")["mapped_lesion_main_term_ID"].to_dict()
        normal_lesion_dict = load_normal_entries(as_dict=True)

        display(HTML("<p>identified mappings for organ and lesion terms are now applied to the dataset...<p>"))

        # apply lesion and organ mapping       
        
        df["harmonized_organ"] = df[f"original {self.organ_column}"].map(df_organ_dict_name).fillna("WHOLE BODY")
        df["harmonized_organ_id"] = df[f"original {self.organ_column}"].map(df_organ_dict_id)

        # apply mapping of NORMAL entries
        df["harmonized_lesion"] = df[f"original {self.lesion_column}"].map(normal_lesion_dict)
        df["harmonized_lesion"] = df[f"original {self.lesion_column}"].map(df_lesion_dict_name).fillna("NORMAL")
        df["harmonized_lesion_id"] = df[f"original {self.lesion_column}"].map(df_lesion_dict_id)
        
        display(HTML("<h3>Successfully applied the harmonization.</h3>"))
        
        lesion_normal_before = df[df[self.lesion_column] == "NORMAL"].shape[0]
        lesion_normal_after = df[df[f"harmonized_lesion"] == "NORMAL"].shape[0]
        
        organ_normal_before = df[df[f"original {self.organ_column}"].isna()].shape[0]
        organ_normal_after = df[df["harmonized_organ"] == "WHOLE BODY"].shape[0]
        
        display(HTML("""<h4>Note: terms that were not assigned to a ontology term are filled with "NORMAL" in case of lesion terms or "WHOLE BODY" in case of organ terms.<h4>"""))
        
        summary_html = f"""
        <p>
        For control: <br>
        Number of NA / "NORMAL" entries in the lesion column before harmonization and before adding back "NORMAL" entries: {lesion_normal_before}<br>
        Number of NA / "NORMAL" entries in the lesion column after harmonization and after adding back "NORMAL" entries: {lesion_normal_after}<br>
        Number of NA entries in the organ column before harmonization and before adding back "NORMAL" entries: {organ_normal_before}<br>
        Number of NA / "WHOLE BODY" entries in the organ column after harmonization and after adding back "NORMAL" entries: {organ_normal_after}
        </p>
        """
        display(HTML(summary_html))
        
        return df


    def explore_harmonized_df(self, 
                    df, 
                    reduce_organ_panel : bool = False,
                    threshold_organ_panel : float = 0.5,
                    remove_animals_with_few_organs : bool = False,
                    threshold_organs_per_animal : int = 5
                    ): 
        """
        Explores and optionally filters the harmonized dataset based on organ coverage and animal completeness.

        This method:
        - Displays the distribution of harmonized organs across animals.
        - Optionally filters out organs studied in fewer than a specified proportion of animals.
        - Optionally removes animals with fewer than a specified number of organs investigated.

        Parameters:
        -----------
        df : pd.DataFrame
            The harmonized dataset containing at least "original MISPEC" and "harmonized_organ" columns.
        reduce_organ_panel : bool, default=False
            If True, removes organs that appear in fewer animals than the specified threshold.
        threshold_organ_panel : float, default=0.5
            Minimum proportion of animals in which an organ must appear to be retained (e.g., 0.5 = 50%).
        remove_animals_with_few_organs : bool, default=False
            If True, removes animals that have fewer than the specified number of organs investigated.
        threshold_organs_per_animal : int, default=5
            Minimum number of organs an animal must have to be retained in the dataset.

        Returns:
        --------
        pd.DataFrame
            The filtered DataFrame after applying the specified organ and animal thresholds.
        """        
        
        # 1. distribution of studied organs
        display(HTML("<h1>Organ Distribution</h1>"))
        
        mi_dataset_organs = df[[self.sample_column, "harmonized_organ"]]
        mi_dataset_organs = mi_dataset_organs.pivot_table(index=self.sample_column, columns="harmonized_organ", aggfunc=lambda x:1, fill_value=0)

        organ_entries = pd.DataFrame(mi_dataset_organs.sum().sort_values(ascending=False)/mi_dataset_organs.shape[0]).reset_index()
        animal_entries = pd.DataFrame(mi_dataset_organs.T.sum().sort_values(ascending=False)).reset_index()

        display(HTML("<h3>Frequency of studied organs across all animals:</h3>"))
        display(organ_entries)
        
        plt.figure(figsize=(12, 10))
        sns.barplot(x="value", y="harmonized_organ", data=organ_entries.rename(columns={0:"value"}))
        plt.xlabel("Value")
        plt.ylabel("Harmonized Organ")
        plt.title("Frequency of Harmonized Organs")
        plt.tight_layout()
        plt.show()
        
        organs_to_keep = organ_entries["harmonized_organ"][organ_entries[0] >= threshold_organ_panel]

        display(HTML(f"<h3>There are {organs_to_keep.shape[0]} organs that were studied in >= {threshold_organ_panel*100} of animals:</h3>"))
        display(organs_to_keep)
        
        animals_with_few_organs = animal_entries[animal_entries[0]<threshold_organs_per_animal][self.sample_column]
        display(HTML(f"""<p>Total number of animals: {animal_entries.shape[0]}<br>
                         Animals with <{threshold_organs_per_animal} organs investigated: {animals_with_few_organs.shape[0]}</p>"""))

        # 2. sort out organs that occur in dataset less then threshold:
        if reduce_organ_panel:
            display(HTML("<h1>Reduction of Organ Panel</h1>"))
            
            display(HTML(f"<h3>Organs that were investigated in <{threshold_organ_panel*100}% of animals are now removed from the dataset...</h3>"))
            display(HTML(f"""<p>df shape before sorting out organs that were investigated in <{threshold_organ_panel*100}% of animals: {df.shape}<br>
                            number of animals before sorting out organs that were investigated in <{threshold_organ_panel*100}% of animals: {df[self.sample_column].drop_duplicates().shape}</p>"""))
            df = df[df["harmonized_organ"].isin(organs_to_keep)]

            display(HTML(f"""<p>df shape after sorting out organs that were investigated in <{threshold_organ_panel*100}% of animals: {df.shape}<br>
                            number of animals after sorting out organs that were investigated in <{threshold_organ_panel*100}% of animals: {df[self.sample_column].drop_duplicates().shape}</p>"""))
        
        # 3. sort out animals with few organs:
        if remove_animals_with_few_organs:
            display(HTML("<h1>Removal of Animals with few organs investigated</h1>"))
            
            display(HTML(f"<h3>Animals with <{threshold_organs_per_animal} organs investigated are now removed from the dataset...</h3>"))
            display(HTML(f"""<p>df shape before sorting out animals with <{threshold_organs_per_animal} organs investigated: {df.shape} <br>
                            number of animals before sorting out animals with <{threshold_organs_per_animal} organs investigated: {df[self.sample_column].drop_duplicates().shape}</p>"""))
            
            df = df[~df[self.sample_column].isin(animals_with_few_organs)].drop_duplicates()
            
            display(HTML(f"""<p>df shape after sorting out animals with <{threshold_organs_per_animal} organs investigated: {df.shape}<br>
                            <p>number of animals after sorting out animals with <{threshold_organs_per_animal} organs investigated: {df[self.sample_column].drop_duplicates().shape}</p>"""))
            
        return df

