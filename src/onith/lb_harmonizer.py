import pandas as pd
import numpy as np
import os
from IPython.display import HTML, display, clear_output
import seaborn as sns
import matplotlib.pyplot as plt
from .harmonizer_base import *
from .ontology_utils import *
import subprocess
import json
from .export_logger import *
import ipywidgets as widgets
import warnings
from sklearn.impute import KNNImputer
from typing import Callable
from pathlib import Path


@dataclass
class LBHarmonizer(HarmonizerBase):
    original_marker_term_column: str = "LBTESTCD"
    measurement_column: str = "marker_name"
    value_column: str = "LBSTRESC"
    unit_column: str = "LBSTRESU"
    day_column: str = "LBDY"
    specimen_column: str = "LBSPEC"


    def automatic_mapping_lb(self, df, metadata_path: str = None) -> pd.DataFrame:
        """
        Automatically maps marker terms to standardized terminology using a predefined dictionary (based on SEND CDISC Terminology).

        Parameters:
        -----------
            df (pd.DataFrame): Input DataFrame containing marker names.
            metadata_path (str, optional): Path to metadata file used for filtering. Defaults to None.

        Returns:
        -----------
            pd.DataFrame: DataFrame with harmonized marker terms and corresponding marker IDs.

        Workflow:
        -----------
            1. Logs the original shape of the input DataFrame.
            2. Filters the DataFrame using metadata (if provided).
            3. Loads a dictionary mapping synonyms to standardized marker names and IDs.
            4. Retains only relevant columns.
            5. Maps original marker terms to standardized names and IDs.
            6. Displays any marker terms that could not be mapped.
        """

        original_shape = df.shape
        print("df shape after import: ", original_shape)

        # reduce by metadata
        df = self.filter_by_metadata(df, output_dir=self.output_dir, sample_column=self.sample_column, project_name=self.project_name, metadata_path=metadata_path)

        # load marker dictionary
        lb_dict = load_and_prepare_dict(input_type="lb")
        dict_synonym_to_name = lb_dict.set_index("synonym")["name"].to_dict()
        dict_name_to_id = lb_dict.set_index("name")["id"].to_dict()

        # only keep relevant columns
        df = df[[self.sample_column, self.original_marker_term_column, self.value_column, self.unit_column, self.specimen_column, self.day_column, self.study_id_column]].drop_duplicates()

        # apply dictionary to harmonize marker names
        display(HTML("<p>Harmonization of blood marker terms is now running...</p>"))
        df[self.measurement_column] = df[self.original_marker_term_column].map(dict_synonym_to_name)
        df["marker_id"] = df[self.original_marker_term_column].map(dict_name_to_id)

        # identify marker names not found in the dictionary
        not_found_marker_terms = df[df[self.measurement_column].isna()][self.original_marker_term_column].drop_duplicates()

        display(HTML(f"""
        <h1>Checkpoint: Are any blood markers lost in translation?</h1>
        <h3>Blood marker terms not found in the terminology:</h3>"""))
        display(not_found_marker_terms)
        
        return df


    def jsons_for_streamlit_lb(self, df):
        """
        Converts the dataFrame containing the mapping candidates into a list of dictionaries formatted for Streamlit display and saves them as a JSON file for use in the manual mapping editor.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing original terms and mapping candidates (as lists or strings).

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
        
        # retrieve dropdown list from marker dictionary
        lb_dict = load_and_prepare_dict("lb")
        drop_down_list = lb_dict["name"].drop_duplicates().to_list()
        
        # retrieve term_list from df (for row-specific term order in each drop down menu of the manual mapping editor)
        term_list = df[df[self.measurement_column].isna()][self.original_marker_term_column].drop_duplicates().to_list()
        
        manual_mappings = []
        for term in term_list:
            manual_mappings.append({
                "Original Term": term,
                "Row Term Order": drop_down_list
            })
        
        input_type = "lb"
        
        json_path_mapping = os.path.join(self.temp_dir, f"{self.project_name}_{input_type}_terms_for_manual_mapping.json")
        
        with open(json_path_mapping, "w") as f:
            json.dump(manual_mappings, f)
            
        display(HTML(f"<p>The input for the manual mapping editor was saved to: {json_path_mapping}</p>"))
        
        return json_path_mapping


    def launch_streamlit_lb(self, sharable, json_path_mapping, input_type, json_path_dict, show_term_info):
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
        - Configures the default path for saving progress as: `{self.project_name}_{input_type}_progress_manual_mapping.json` in `self.temp_dir`. Can be adjusted in the application.
        """
        
        module_dir = os.path.dirname(__file__)  
        app_path = os.path.join(module_dir, "manual_mapping_editor.py")
        json_path_mapping = json_path_mapping
        json_path_dict = json_path_dict
        save_dir = os.path.join(self.temp_dir, f"{self.project_name}_{input_type}_progress_manual_mapping.json")
        subprocess.Popen(["streamlit", "run", app_path, json_path_mapping, json_path_dict, save_dir, sharable, show_term_info])
        display(HTML(f"<p>The editor will now open (as new tab in your internet explorer, can take a moment)...</p>"))
        
        return 
    

    def launch_manual_mapping_lb(self, df, sharable : str = "False") -> None:
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

        input_type = "lb"
        show_term_info : str = "False"      
        json_path_dict = ""
        
        json_path_mapping = self.jsons_for_streamlit_lb(df)
        self.launch_streamlit_lb(sharable, json_path_mapping, input_type, json_path_dict, show_term_info)
    
        return
    

    def add_new_synonyms_to_obo_lb(self, obo_file_path, mapping_man):
        """
        Adds new manually reviewed synonyms (as exported from the manual mapping editor and found by the fuzzy matching) to the respective local OBO ontology file.

        This method:
        - Parses the OBO file while preserving its structure.
        - Identifies terms in the ontology by their IDs.
        - Adds new synonyms to the appropriate term blocks in the OBO file.
        - Updates the header with a timestamp indicating the last manual completion.
        - Adds term specific timestamps for version control
        """
        
        with open(obo_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # identify the header section in lb terminology
        header_end_index = next(i for i, line in enumerate(lines) if line.strip().startswith("[Term]"))

        # generate manual completion timestamp
        current_date = datetime.now().strftime("%Y-%m-%d")
        manual_completion_entry = f"last manual completion with additional synonyms: {current_date}\n"

        header_lines = lines[:header_end_index]
        if any("last manual completion with additional synonyms:" in line for line in header_lines):
            header_lines = [manual_completion_entry if "last manual completion with additional synonyms:" in line else line for line in header_lines]
        else:
            header_lines.append(manual_completion_entry)

        # parse [Term] blocks (marks each new term)
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
        
        # build a mapping from id to term block index
        id_to_index = {}
        for idx, term in enumerate(terms):
            for line in term:
                if line.strip().startswith("id:"):
                    term_id = line.strip().split("id:")[1].strip()
                    id_to_index[term_id] = idx
                    break

        # add synonyms to the correct term blocks, but only if the synonym does'nt already exist
        for _, row in mapping_man.iterrows():
            term_id = row[f"marker_id"]
            original_synonym = row[self.original_marker_term_column]
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
                    continue  # skip if synonym already exists

                # find all synonym lines
                synonym_indices = [i for i, l in enumerate(term) if l.strip().startswith("synonym:")]

                # insert after last synonym
                if synonym_indices:
                    insert_at = synonym_indices[-1] + 1
                    term.insert(insert_at, new_synonym)
                else:
                    blank_indices = [i for i, l in enumerate(term) if l.strip() == ""]
                    insert_at = blank_indices[0] if blank_indices else len(term)
                    term.insert(insert_at, new_synonym)

        # reconstruct the updated file
        updated_dict = header_lines + ["\n"] + [line for term in terms for line in term]

        return updated_dict


    def apply_mapping_lb(self, 
                        df: pd.DataFrame,
                        integrate_manual_mapping: bool,
                        delete_unmapped: bool,
                        update_dict: bool,
                        add_specimen: bool,
                        json_path_mapping: str = None,
                        json_path_progress: str = None) -> pd.DataFrame:
        """
        Applies automatic and manual mapping to harmonize marker terms in the LB domain.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing LB domain data.
            integrate_manual_mapping (bool): Whether to integrate manually mapped terms from a JSON file.
            delete_unmapped (bool): If True, removes entries with unmapped marker terms.
            update_dict (bool): If True, updates the local terminology dictionary with new synonyms.
            add_specimen (bool): If True, appends specimen information to the marker name.
            json_path_mapping (str, optional): Path to JSON file containing metadata for manual mapping.
            json_path_progress (str, optional): Path to JSON file containing user selections from manual mapping.

        Returns:
            pd.DataFrame: DataFrame with applied mappings and optional updates.

        Description:
            - Loads the LB terminology dictionary and maps marker terms to standardized names and IDs.
            - Optionally integrates manual mappings from interactive mapping tools.
            - Handles unmapped terms based on user preference (delete or retain original term).
            - Saves the mapped DataFrame.
            - Optionally updates the local ontology with new synonyms.
            - Optionally appends specimen information to marker names for clarity.

        Notes:
            - Requires prior execution of `automatic_mapping_lb` to initialize mapping columns.
        """

        input_type = "lb"
        
        # fetch marker id 
        lb_dict = load_and_prepare_dict(input_type="lb")
        dict_name_to_id = lb_dict.set_index("name")["id"].to_dict()

        if integrate_manual_mapping:
            
            # set paths for metadata (input for streamlit) and selections (as selected from the dropdown menu by the user and then saved as progress) 
            if json_path_mapping is None:
                json_path_mapping = os.path.join(self.temp_dir, f"{self.project_name}_{input_type}_terms_for_manual_mapping.json")
            if json_path_progress is None:
                json_path_progress = os.path.join(self.temp_dir, f"{self.project_name}_{input_type}_progress_manual_mapping.json")
                    
            # load metadata of manually mapped terms
            if not os.path.exists(json_path_mapping):
                raise FileNotFoundError(f"JSON file not found: {json_path_mapping}")
            
            with open(json_path_mapping, "r") as f:
                mapping_man = json.load(f)
            
            mapping_man = pd.DataFrame(mapping_man)
            mapping_man = mapping_man.drop(columns="Row Term Order")

            mapping_man = mapping_man.rename(columns={"Original Term": self.original_marker_term_column})
            
            # load term selections from the manual mapping editor
            if not os.path.exists(json_path_progress):
                raise FileNotFoundError(f"JSON file not found: {json_path_progress}")

            with open(json_path_progress, "r") as f:
                term_selection = json.load(f)
                
            term_selection = term_selection.get("selections")

            # merge term selections with metadata of manually mapped terms                
            mapping_man[self.measurement_column] = term_selection

            mapping_man["marker_id"] = mapping_man[self.measurement_column].map(dict_name_to_id)
            
            mapping_man_dict = mapping_man.set_index(self.original_marker_term_column)[self.measurement_column].to_dict()

            # merge manually with automatically mapped terms
            df = df.copy()
            mask = df[self.measurement_column].isna()
            df.loc[mask, self.measurement_column] = df.loc[mask, self.original_marker_term_column].map(mapping_man_dict)
        
        df.loc[:, "marker_id"] = df.loc[:, self.measurement_column].map(dict_name_to_id)

        if delete_unmapped == True: 
            print(f"the parameter 'delete_unmapped' was set to True. Shape of the df before deleting unmapped marker entries: {df.shape}")
            df = df.dropna(subset=self.measurement_column)
            print(f"Shape of the df after deleting unmapped marker entries: {df.shape}")
        
        # if unmapped marker names shall not be deleted, their name is copied to marker_name instead
        elif delete_unmapped == False:
            df.loc[df[self.measurement_column].isna(), self.measurement_column] = df.loc[df[self.measurement_column].isna(), self.original_marker_term_column]

        # save mapping results
        mapping_path = os.path.join(self.output_dir, f"{self.project_name}_{input_type}_df_after_mapping.csv")
        df.to_csv(mapping_path) 

        display(HTML("<p><strong>These are the final results of the mapping:</strong></p>"))
        display(df[[self.original_marker_term_column,self.measurement_column]].drop_duplicates())
        display(HTML(f"<p>The dataframe with the applied mapping is saved to {mapping_path}</p>"))

        # update lesion_dict with new synonyms if wished
        if update_dict == True & integrate_manual_mapping == True:
            # add mapped original terms to as dict syononyms
            base_path = Path(__file__).resolve().parent / "dict"
            obo_file_path = os.path.join(base_path,"lb_terminology.obo")
                
            updated_dict = self.add_new_synonyms_to_obo_lb(obo_file_path, mapping_man)
            
            # convert back to obo and save
            subfolder = save_updated_dict(obo_file_path, updated_dict, input_type)
            
            display(HTML(f"<p>Your local copy of the ontology was updated with the mapped terms from your dataset and saved to: onith/dict/lb_terminology.obo </p>"))
            display(HTML(f"<p>The previous local version of the ontology can be restored from: {subfolder}</p>"))
            
        if add_specimen:
            df[self.measurement_column] = df.apply(
                lambda row: row[self.measurement_column]
                if str(row[self.measurement_column]).endswith(f" - {row[self.specimen_column]}")
                else f"{row[self.measurement_column]} - {row[self.specimen_column]}",
                axis=1
            )

            display(HTML("<p>The specimen information was added to the marker name to avoid confusion. The current state of the dataset:</p>"))
            display(df)

        return df
    

    def interactive_panel_selector(self, rel_frq: pd.DataFrame) -> callable:
        """
        Launches an interactive widget-based panel selector for the LB markers based on their relative frequency.

        Parameters:
            rel_frq (pd.Series or pd.DataFrame): A Series or DataFrame with marker names as index and
                                                relative frequencies as values.

        Returns:
            function: A callable `select_marker_panel()` that returns the selected markers as a DataFrame.

        Description:
            - Displays an interactive threshold slider to preselect markers based on frequency.
            - Allows manual selection/deselection of markers via checkboxes.
            - Provides a scrollable table view of all markers and their relative frequencies.
            - Includes a submit button to make the selection available for subsequent steps.
            - Appends specimen information to marker names if configured in the harmonization pipeline.

        Notes:
            - This tool supports the selection of a consistent marker panel across studies.
            - Intended for use in Jupyter Notebook.
            - The selected panel can be retrieved by calling the returned `select_marker_panel()` function.
        """

        df = pd.DataFrame(rel_frq)
        df = df.reset_index()

        # Set up widgets
        threshold_slider = widgets.FloatSlider(
            value=0.9, min=0, max=1.0, step=0.01, description="Threshold:", continuous_update=False
        )

        checkboxes = [widgets.Checkbox(description="", indent=False) for _ in range(len(df))]

        out = widgets.Output()
        result_out = widgets.Output()
        note_out = widgets.Output()

        # Display note once at the start
        with note_out:
            clear_output()
            display(HTML("<p>Note: In order to retrieve your selection in the next step, click the Submit button.</p>"))

        def update_table(threshold):
            # Update checkboxes based on threshold
            for i, freq in enumerate(df["relative_frequency"]):
                checkboxes[i].value = freq >= threshold

            # Define consistent widths
            marker_name_width = "600px"
            relative_freq_width = "150px"
            checkbox_width = "100px"

            # Build table header
            header = widgets.HBox([
                widgets.Label(self.measurement_column, layout=widgets.Layout(width=marker_name_width)),
                widgets.Label("relative_frequency", layout=widgets.Layout(width=relative_freq_width)),
                widgets.Label("marker_panel", layout=widgets.Layout(width=checkbox_width))
            ])

            # Build table rows
            rows = []
            for i, row in df.iterrows():
                row_widget = widgets.HBox([
                    widgets.Label(str(row[self.measurement_column]), layout=widgets.Layout(width=marker_name_width)),
                    widgets.Label(f"{row['relative_frequency']:.3f}", layout=widgets.Layout(width=relative_freq_width)),
                    checkboxes[i]
                ])
                rows.append(row_widget)

            # Combine header and rows into one VBox
            table_content = widgets.VBox([header] + rows)

            # Wrap the table in a scrollable container
            scrollable_table = widgets.Box([table_content], layout=widgets.Layout(
                overflow_y="auto",
                height="400px",
                border="1px solid lightgray",
                padding="5px",
                display="block"
            ))

            # Display the scrollable table (clear output first)
            with out:
                clear_output()
                display(scrollable_table)

        def on_threshold_change(change):
            update_table(change["new"])

        # Attach observer only once
        threshold_slider.observe(on_threshold_change, names="value")

        # Submit button to make selection available in the next step
        submit_button = widgets.Button(description="Submit Selection", button_style="success")

        marker_panel = pd.DataFrame()  # Placeholder

        def on_submit_clicked(b):
            nonlocal marker_panel
            selected_rows = [i for i, cb in enumerate(checkboxes) if cb.value]
            marker_panel = df.iloc[selected_rows].copy()
            with result_out:
                clear_output()
            with note_out:
                clear_output()
                display(HTML("<p style='color:green;'>Selection submitted. You can now proceed to the next step.</p>"))

        # Attach click handler only once
        submit_button.on_click(on_submit_clicked)

        # Initial table display
        update_table(threshold_slider.value)

        # Display all widgets only once
        display(widgets.VBox([threshold_slider, out, note_out, submit_button, result_out]))

        # Return function to retrieve selected DataFrame
        def select_marker_panel():
            return marker_panel

        return select_marker_panel


    def explore_marker_frequency(self, df: pd.DataFrame, rel_freq_group: Literal["sample", "study"]) -> pd.Series:
        """
        Visualizes and calculates the relative frequency of LB markers across samples or studies.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing LB domain data.
            rel_freq_group (Literal["sample", "study"]): Determines the grouping level for frequency calculation.
                - "sample": Frequency is calculated per animal/sample.
                - "study": Frequency is calculated per study.

        Returns:
            pd.Series: A Series with marker names as index and their relative frequency as values.

        Description:
            - Drops duplicate marker measurements per sample or study to avoid overcounting.
            - Computes the relative frequency of each marker across the selected group.
            - Generates a horizontal scrollable bar plot showing marker frequencies.
            - Saves the plot as a PNG image and displays it using IPython widgets.

        Notes:
            - Helps identify consistently measured markers across samples or studies.
            - Supports downstream marker panel selection and harmonization decisions.
            - Intended for use in Jupyter environments with IPython widgets.
        """

        if rel_freq_group == "sample":
            rel_freq_col = self.sample_column
        else:
            rel_freq_col = self.study_id_column

        # drop duplicate measurements per animal-marker pair
        unique_measurements = df.drop_duplicates(subset=[rel_freq_col, self.measurement_column])

        # calculate the relative frequency of each marker
        rel_frq = unique_measurements[self.measurement_column].value_counts(normalize=False)/df[rel_freq_col].drop_duplicates().shape[0]
        rel_frq.name = "relative_frequency"

        plt.figure(figsize=(40, 6))
        rel_frq.sort_values(ascending=False).plot(kind="bar", color="skyblue", edgecolor="black")

        plt.xlabel("Marker Name", fontsize=12)
        plt.ylabel("Relative Frequency", fontsize=12)
        plt.title("In How Many Animals Was This Marker Measured?", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # save plot to image file
        plot_filename = os.path.join(self.temp_dir, f"{self.project_name}_lb_marker_frequency_plot.png")
        plt.savefig(plot_filename, bbox_inches="tight")

        plt.close()

        with open(plot_filename, "rb") as f:
            image_bytes = f.read()

        img_widget = widgets.Image(
            value=image_bytes,
            format="png",
            layout=widgets.Layout(
                width="auto",
                height="auto"
            )
        )

        scrollable = widgets.Box(
            [img_widget],
            layout=widgets.Layout(
                overflow_x="auto",
                overflow_y="hidden",
                width="3000px",
                border="1px solid gray",
                display="block",
                flex_flow="row",
            )
        )

        display(HTML("<h3>This is the relative frequency in how many animals the different markers were measured at least once."))
        display(scrollable)
        display(HTML("<p>Use this information to decide on a threshold for your cross-study analysis: What is the minimum relative frequency of a marker to be part of the marker panel of your analysis?"))
        
        return rel_frq


    def define_marker_panel(self, df: pd.DataFrame, rel_freq_group: Literal["sample", "study"]):
        """
        Launches an interactive interface to define a marker panel based on relative frequency.

        Parameters:
            df (pd.DataFrame): Input DataFrame with harmonized LB data.
            rel_freq_group (Literal["sample", "study"]): Grouping level for frequency calculation.

        Returns:
            function: A callable that returns the selected marker panel as a DataFrame.
        """
        
        rel_frq = self.explore_marker_frequency(df, rel_freq_group)
        display(HTML("<h3>Select the markers that you want to include in your analysis:"))
        
        select_marker_panel = self.interactive_panel_selector(rel_frq)
        
        return select_marker_panel


    def apply_marker_panel(self, df: pd.DataFrame, select_marker_panel) -> pd.DataFrame:
        """
        Filters the DataFrame to include only markers from the selected panel.
        """

        marker_panel = select_marker_panel()
        print("df shape before reducing to marker panel:", df.shape)
        df = pd.merge(df, marker_panel[self.measurement_column], on=self.measurement_column, how="inner")
        print("df shape after reducing to marker panel:", df.shape)
        
        return df
    
        
    def explore_nonnumeric(self, df):
        """
        Identifies and reports non-numeric entries in a specified value column of a DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data to be analyzed.

        Returns:
            None: Displays an HTML report indicating whether non-numeric values are present,
                their frequency relative to each measurement category, and prompts for
                further action if necessary.

        Attributes used:
            self.value_column (str): Name of the column expected to contain numeric values.
            self.measurement_column (str): Name of the column used to group the data for frequency analysis.

        Behavior:
            - Converts the value column to numeric, coercing errors to NaN.
            - Identifies rows where conversion failed (i.e., non-numeric entries).
            - If no non-numeric values are found, displays a green message.
            - If non-numeric values are found:
                - Lists unique non-numeric entries.
                - Calculates their relative frequency within each measurement group.
                - Displays an HTML report with findings and a prompt for imputation strategy.
        """

        # create mask for non-numeric entries
        non_numeric_mask = pd.to_numeric(df[self.value_column], errors="coerce").isna()
        non_numeric_df = df[non_numeric_mask]

        # grouping and frequency calculation (only if non-numeric values exist)
        if non_numeric_df.empty:
            show_green_message = True
        else:
            total_testcd = df.groupby(self.measurement_column).size()
            non_numeric_entries = non_numeric_df[[self.measurement_column, self.value_column]].value_counts().divide(total_testcd)
            show_green_message = non_numeric_entries.empty

        if show_green_message:
            html_output = """
            <p style="color:green;">No non-numeric values found in the dataset.</p>
            """
        else:
            non_numeric_entries_list = non_numeric_df[self.value_column].drop_duplicates().to_list()
            html_output = f"""
            <h1>Checkpoint: Is it ok to convert all non-numerical entries to NaN? Are they missing at random?</h1>
            <h2>Following non-numerical STRESC entries occur in the dataset:</h2>
            <p>{non_numeric_entries_list}</p>
            <p-> Decide on imputation strategy. Are they all missing at random?</p>
            <p>Frequency of non-numerical entries, relative to the total number of entries for respective testcd:</p> 
            {non_numeric_entries.to_frame().to_html()}
            <p><strong>ToDo:</strong> If it is ok to convert all those non-numerical entries to NaN, no action is required. Otherwise, handle them now before continuing.</p>
            """

        display(HTML(html_output))

        return 
    

    def check_units(self, row: pd.Series, df: pd.DataFrame) -> bool:
        """
        Checks if at least 90% of unit entries for a marker in a study match the expected units.
        """

        testcd = row[self.measurement_column]
        studyid = row[self.study_id_column]
        expected_units = row[self.unit_column]

        # filter for marker and study
        filtered_df = df[(df[self.measurement_column] == testcd) & (df[self.study_id_column] == studyid)]

        # group by subject and day, check if all expected units are present
        unit_check = filtered_df.groupby([self.sample_column, self.day_column])[self.unit_column].apply(
            lambda x: all(unit in x.values for unit in expected_units)
        )

        percentage = unit_check.mean()
        return percentage >= 0.9


    def explore_and_harmonize_units(self, df, unit_dict=None):
        """
        Explore and harmonize units used for blood markers across studies.

        This function performs the following:
        1. Replaces unit terms using a provided or default unit dictionary.
        2. Identifies blood markers and studies where multiple units are used.
        3. Checks whether both units are consistently present across samples and timepoints.
        4. Converts percentage-based hematology values to absolute values using WBC counts.
        5. Displays HTML summaries to guide the user through the harmonization process.
        6. Returns a cleaned and harmonized DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing blood marker data.
            unit_dict (dict, optional): A dictionary mapping unit variants to standardized terms.

        Returns:
            pd.DataFrame: The harmonized DataFrame.
        """
        
        # Step 1: Harmonize unit terms
        display(HTML(f"<h1>Checkpoint: Do any units have to be converted before we continue?</h1>"))
        display(HTML("<p>Step 1: Searching for unit term synonyms...</p>"))
        if unit_dict is None:
            unit_dict = load_unit_dict()
        df = df.replace({self.unit_column: unit_dict})

        # Step 2: Identify blood markers with multiple units
        display(HTML("<p>Step 2: Exploring unit usage across markers and studies...</p>"))
        units = pd.DataFrame(df[[self.measurement_column, self.unit_column]].value_counts()).sort_values(by=self.measurement_column)
        multiple_units = units.groupby(self.measurement_column).filter(lambda x: len(x) > 1)

        studies_with_multiple_units = df.groupby([self.study_id_column, self.measurement_column])[self.unit_column].nunique().reset_index()
        studies_with_multiple_units = studies_with_multiple_units[studies_with_multiple_units[self.unit_column] > 1]

        duplicated_units = df.dropna(subset=[self.unit_column]).groupby([self.study_id_column, self.measurement_column])[self.unit_column].unique().reset_index()
        duplicated_units = duplicated_units[duplicated_units.apply(lambda x: len(x[self.unit_column]) > 1, axis=1)]

        # Step 3: Check if all units are consistently present
        duplicated_units["All units present for at least 90% of samples and timepoints"] = duplicated_units.apply(self.check_units, axis=1, args=(df,))
        duplicated_units_false = duplicated_units[duplicated_units["All units present for at least 90% of samples and timepoints"] == False]

        # Step 4: Display findings
        display(HTML(f"""<h3>Blood markers with different units used across and within studies: </h3><br>
                    <p>-> if different units were only used across different studies, but never within a study, it can potentially remain unresolved as long as the units are interchangable (so that the relative statistics are still equivalent)</p>"""))
        display(HTML(multiple_units.to_html()))

        if not duplicated_units.empty:
            display(HTML("""<p style='color:red;'>Warning: There are studies where multiple units were used for the same blood marker.<br>
                        -> has to be resolved before calculating relative statistics</p>"""))
            display(HTML(duplicated_units.to_html()))

        # Step 5: Harmonize values
        display(HTML("<p>Step 3: Converting units and removing redundant entries if needed...</p>"))
        df[self.value_column] = pd.to_numeric(df[self.value_column], errors="coerce")

        duplicated_units_true = duplicated_units[duplicated_units["All units present for at least 90% of samples and timepoints"] == True]
        condition = (
            df[self.study_id_column].isin(duplicated_units_true[self.study_id_column]) &
            df[self.measurement_column].isin(duplicated_units_true[self.measurement_column]) &
            df[self.unit_column].isin(["%", "10^9/L", "10^12/L"])
        )
        df = df[~condition]

        hematology_marker = duplicated_units_false[self.measurement_column].drop_duplicates().to_list()

        df["USUBJID_DY"] = df[self.sample_column] + "_" + df[self.day_column].astype(str)
        df["value original"] = df[self.value_column].copy()
        df["unit original"] = df[self.unit_column].copy()

        wbc_value_dict = df[df[self.measurement_column] == "WHITE BLOOD CELLS"].set_index("USUBJID_DY")[self.value_column].to_dict()
        wbc_unit_dict = df[df[self.measurement_column] == "WHITE BLOOD CELLS"].set_index("USUBJID_DY")[self.unit_column].to_dict()

        df["WBC_value"] = df["USUBJID_DY"].map(wbc_value_dict)
        df["WBC_unit"] = df["USUBJID_DY"].map(wbc_unit_dict)

        df["value_abs"] = df["WBC_value"].astype(float) * df[self.value_column].astype(float) / 100

        for index, row in df.iterrows():
            if row[self.measurement_column] in hematology_marker and row[self.unit_column] == "%":
                df.at[index, self.value_column] = row["value_abs"]
                df.at[index, self.unit_column] = row["WBC_unit"]

        df = df.drop(columns=["WBC_value", "WBC_unit", "value_abs", "USUBJID_DY"])
        df = df.drop_duplicates()

        # Step 6: Final check
        display(HTML("<p>Final check: Re-evaluating unit consistency after harmonization...</p>"))
        units_final = pd.DataFrame(df[[self.measurement_column, self.unit_column]].value_counts()).sort_values(by=self.measurement_column)
        duplicated_units_final = df.dropna(subset=[self.unit_column]).groupby([self.study_id_column, self.measurement_column])[self.unit_column].unique().reset_index()
        duplicated_units_final = duplicated_units_final[duplicated_units_final.apply(lambda x: len(x[self.unit_column]) > 1, axis=1)]

        if not duplicated_units_final.empty:
            display(HTML("""<p style='color:red;'>Warning: There are still studies where multiple units were used for the same blood marker.<br>
                        -> Please resolve these before proceeding.</p>"""))
            display(HTML(duplicated_units_final.to_html()))
        else:
            display(HTML(f"<p style='color:green;'>All unit conflicts are resolved. You can now proceed.</p>"))

        print("df shape after unit harmonization: ", df.shape)
        
        return df


    def control_stats(self, df: pd.DataFrame, metadata_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes control group statistics and separates treated animals for further analysis.

        Parameters:
            df (pd.DataFrame): Input DataFrame with harmonized LB data.
            metadata_path (str, optional): Path to metadata file. If None, uses default project path.

        Returns:
            tuple: (control_stats, treated) where:
                - control_stats (pd.DataFrame): Mean and standard deviation of control animals per marker and study.
                - treated (pd.DataFrame): Subset of treated animals for downstream analysis.
        """

        if metadata_path is None:
            metadata_path = os.path.join(self.output_dir, f"metadata_{self.project_name}.csv")
            metadata = pd.read_csv(metadata_path)
        else: 
            metadata = pd.read_csv(metadata_path)
        
        # merge with metadata
        df = pd.merge(df, metadata[[self.sample_column,self.group_id_column]], on=self.sample_column, how="inner").drop_duplicates()
        print("merged with metadata: ", df.shape)
        
        # convert any nonnumeric to nan, drop nan
        df[self.value_column] = pd.to_numeric(df[self.value_column], errors="coerce")
        df.dropna(subset=self.value_column, inplace=True)
        df[self.value_column] = df[self.value_column].astype(float)
        print("removing nonnumeric and NaN: ", df.shape)
        
        # unique study id blood marker combinations
        df["study_id_marker"] = df[self.study_id_column] + "_" + df[self.measurement_column]
            
        # filter for controls and treated
        controls = df[[self.sample_column,self.measurement_column,self.value_column, self.study_id_column, "study_id_marker"]][df[self.group_id_column] == 0]
        treated = df[[self.sample_column,self.measurement_column,self.value_column, self.study_id_column, "study_id_marker"]][df[self.group_id_column] != 0]
        print("only control animals: ", controls.shape)
        print("only treated animals: ", treated.shape)

        # calculate mean and st.dev. of controls across all timepoints across all control animals of the study
        control_stats = controls[["study_id_marker", self.value_column]].groupby("study_id_marker")[self.value_column].agg(["mean", "std"])
        control_stats = pd.merge(controls[[self.measurement_column, "study_id_marker",self.study_id_column]], control_stats, on="study_id_marker", how="inner").drop_duplicates().sort_values(by=self.study_id_column).reset_index(drop=True)
        print("average all timepoints of all control animals per blood marker: ", control_stats.shape)

        # create HTML output
        html_output = f"""
        <h1>Checkpoint: Does statistics of control animals per study per blood marker look reasonable? Anything suspicious?</h1>
        <h2>control statistics per blood marker: </h2>
        {control_stats.head(40).to_html()}
        <p>If everything seems ok, we can continue by using the mean and standard deviation z-score calculation (and timepoint selection if necessary) of the treated animals.</p>
        """
        
        display(HTML(html_output))   

        return control_stats, treated


    def reduce_timepoints(self, control_stats, treated):
        """
        Reduce timepoints for treated animals based on the largest difference to control means.

        Parameters:
            control_stats (pd.DataFrame): DataFrame containing control statistics (mean and std).
            treated (pd.DataFrame): DataFrame containing treated animals' data (no controls).

        Returns:
            pd.DataFrame: DataFrame with reduced timepoints for treated animals.
        """
        
        warnings.filterwarnings("ignore", message="Clustering large matrix with scipy.*")

        # before reduction
        pivot = pd.pivot_table(
            treated[[self.sample_column, self.measurement_column]],
            index=self.sample_column,
            columns=self.measurement_column,
            aggfunc=len,
            fill_value=0
        )

        def display_clustermap(pivot):
            g = sns.clustermap(pivot, cbar=True, vmax=3, figsize=(10, 10), cmap="viridis")
            g.ax_heatmap.set_xticklabels([])
            g.ax_heatmap.set_yticklabels([])
            g.ax_heatmap.tick_params(left=False, bottom=False, right=False)
            g.ax_heatmap.set_xlabel("Blood Markers")
            g.ax_heatmap.set_ylabel("Animals")  
            g.ax_heatmap.set_title("Number of Measurements per Bloodmarker per Animal")
            
            colorbar = g.ax_heatmap.collections[0].colorbar
            colorbar.set_ticks([0, 1, 2, 3])
            colorbar.set_ticklabels(["0", "1", "2", ">/= 3"]) # always the same range for the color bar to make it better comparable

            plt.show()

        display_clustermap(pivot)

        # add control mean column 
        treated = self.add_control_mean(control_stats, treated)

        # calculate difference to control mean
        treated["diffr"] = abs(treated[self.value_column].astype(float) - treated["control mean"].astype(float))

        # filter for row with largest difference
        treated_filtered = treated.loc[treated.groupby([self.sample_column, self.measurement_column])["diffr"].idxmax()]
        treated_filtered.drop(columns="diffr", inplace=True)
        print("Shape after filtering for rows with largest difference to control means: ", treated_filtered.shape)

        # after reduction
        pivot_filtered = pd.pivot_table(
            treated_filtered[[self.sample_column, self.measurement_column]],
            index=self.sample_column,
            columns=self.measurement_column,
            aggfunc=len,
            fill_value=0
        )

        display_clustermap(pivot_filtered)

        return treated_filtered


    def add_control_mean(self, control_stats: pd.DataFrame, treated: pd.DataFrame) -> pd.DataFrame:
        """
        Merges control group statistics into the treated dataset.
        """

        # add control mean column
        print("Original shape: ", treated.shape)
        treated = pd.merge(
            treated,
            control_stats[["study_id_marker", "mean", "std"]],
            on="study_id_marker",
            how="inner"
        ).rename(columns={"mean": "control mean", "std": "control st. dev."})
        print("Shape after adding column with control means and standard deviations: ", treated.shape)
            
        return treated
    
    
    def calculate_zscore(self, df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
        """
        Calculates z-scores using control mean and standard deviation.
        """

        # epsilon for regularization
        df["z-score"] = (df[self.value_column].astype(float) - df["control mean"].astype(float) + epsilon) / (df["control st. dev."].astype(float) + epsilon)
        
        nan_nr = df["z-score"].isna().sum()
        
        df["z-score"] = df["z-score"].replace({np.inf:np.nan, -np.inf:np.nan})
        
        critical_entries = df[df["z-score"].isna()]
        
        if not critical_entries.empty:
            print("number of NaN: ", nan_nr)
            print("number of inf: ", df["z-score"].isna().sum() - nan_nr)
            display(HTML(critical_entries.to_html())) 
        else:
            display(HTML("<p style='color:green;'> All fine. No NaN or inf values after converting to z-score.</p>"))
        
        return df  


    def pivot_and_sortout(self, df, threshold_sample_nan: float = 0.30, threshold_marker_nan: float = 0.30):
        """
        Creates a pivot table of z-scores and filters out samples and markers with excessive missing values.

        This function performs the following steps:
        1. Generates a pivot table with samples as rows and markers as columns.
        2. Calculates the proportion of missing values per sample and per marker.
        3. Removes samples with missing values above the specified threshold.
        4. Removes markers with missing values above the specified threshold.
        5. Displays HTML summaries of the filtering process and resulting data structure.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing z-score data.
            threshold_sample_nan (float): Maximum allowed proportion of NaN values per sample (default: 0.30).
            threshold_marker_nan (float): Maximum allowed proportion of NaN values per marker (default: 0.30).

        Returns:
            pd.DataFrame: A filtered pivot table with acceptable levels of missing data.

        Attributes used:
            self.sample_column (str): Column name identifying individual samples.
            self.measurement_column (str): Column name identifying measurement types (markers).
            self.study_id_column (str): Column name identifying study IDs.
        """

        lb_pivot = pd.pivot_table(df[[self.sample_column, self.measurement_column, "z-score"]], index=self.sample_column, columns=self.measurement_column, values="z-score", fill_value=np.nan)
        original_shape = lb_pivot.shape
        
        # NaN per animal
        nan_per_animal = lb_pivot.T.isna().sum().sort_values(ascending=False) / lb_pivot.shape[1]
        relative_animals_with_less_nan = lb_pivot.loc[nan_per_animal[nan_per_animal <= threshold_sample_nan].index].shape[0] / lb_pivot.shape[0]
        
        # number of animals per STUDY
        lb_pivot_animals = pd.merge(lb_pivot, df[[self.sample_column, self.study_id_column]], on=self.sample_column, how="inner").drop_duplicates()[self.study_id_column].value_counts()
        
        # removing animals with NaN > threshold
        lb_pivot = lb_pivot.loc[nan_per_animal[nan_per_animal <= threshold_sample_nan].index]
        new_shape_animals = lb_pivot.shape
        
        # number of animals per STUDY remaining
        lb_pivot_animals_remaining = pd.merge(lb_pivot, df[[self.sample_column, self.study_id_column]], on=self.sample_column, how="inner").drop_duplicates()[self.study_id_column].value_counts()
        
        # NaN per marker remaining
        nan_per_marker = lb_pivot.isna().sum().sort_values(ascending=False) / lb_pivot.shape[0]
        
        # removing markers with NaN > threshold
        lb_pivot = lb_pivot[nan_per_marker[nan_per_marker <= threshold_marker_nan].index]
        new_shape_markers = lb_pivot.shape
        
        # display
        html_output = f"""
        <h1>Pivot Table and NaN Analysis</h1>
        <p><strong>Original Shape of Pivot Table:</strong> {original_shape}</p>
        <p><strong>Relative Number of Animals with <= {threshold_sample_nan} NaN Values:</strong> {relative_animals_with_less_nan:.2f}</p>
        <p><strong>Shape of Pivot Table After Removing Animals with > {threshold_sample_nan} NaN:</strong> {new_shape_animals}</p>
        <h3>Number of treated Animals per Study:</h3>
        <p>{lb_pivot_animals.to_frame().to_html()}</p>
        <h3>Number of treated Animals per Study Remaining:</h3>
        <p>{lb_pivot_animals_remaining.to_frame().to_html()}</p>
        <h3>Relative Number of NaN per Marker After Removing Animals with > {threshold_marker_nan} NaN:</h3>
        <p>{nan_per_marker.to_frame().to_html()}</p>
        <p><strong>Shape of Pivot Table After Removing Markers with > {threshold_marker_nan} NaN:</strong> {new_shape_markers}</p>
        """
        
        display(HTML(html_output))
        
        return lb_pivot


    def explore_distribution(self, df: pd.DataFrame) -> None:
        """
        Displays a boxplot to visually assess the distribution and potential outliers.
        """

        display(HTML(f"<h1> Checkpoint: is it necessary to handle outliers? </h1>"))
        sns.boxplot(df)

        plt.xticks(rotation=90)
        plt.show()
        
        return
    
        
    def imputing_lb(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies KNN imputation to fill missing values in the LB pivot table.
        """

        imputer = KNNImputer()

        df_imp = imputer.fit_transform(df)
        df_imp = pd.DataFrame(df_imp, columns=df.columns, index=df.index)
        
        return df_imp
    

    def outlier_removal_and_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes extreme z-score outliers and imputes missing values using KNN.
        """

        print("distribution of NaN before outlier removal: ")
        print(df.isna().sum())
        print("distribution of z-scores before outlier removal: ")
        plt.figure()
        sns.histplot(df.values.flatten(), kde=True)
        plt.show()
        
        # apply zscore threshold for outlier removal
        print("with threshold: abs(z-score) of 10: ", (df.abs() > 10).sum().sum() / (df.shape[0] * df.shape[1])*100, "% of entries will be categorized as outliers and removed.")
        df = df.map(lambda x: x if abs(x) <= 10 else None)
        print("distribution of z-scores after outlier removal: ")
        plt.figure()
        sns.histplot(df.values.flatten(), kde=True)
        plt.show()

        # perform imputation
        df = self.imputing_lb(df)

        display(HTML("<h2>Outlier removal and imputation was performed successfully.</h2>"))
        
        return df
