import nbformat as nbf
import os
from pathlib import Path

example_data_path = Path(__file__).parent / "example_data"

def configure_harmonization_pipeline(notebook_dir:str, om:bool = False, bw:bool = False, lb: bool = False, mi: bool = False):
    """
    Creates a Jupyter notebook that configures and documents a harmonization pipeline for non-clinical SEND data.

    This function generates a structured and interactive Jupyter notebook (`harmonization_pipeline.ipynb`) that guides users
    through the harmonization of various nonclinical data domains (OM, BW, LB, MI). The notebook includes markdown explanations and 
    executable code cells for each selected domain, supporting both automatic and manual term mapping, data cleaning, 
    normalization, feature selection and export of harmonized datasets.

    Parameters:
        notebook_dir (str): Path to the directory where the notebook will be saved. The directory will be created if it doesn't exist.
        om (bool): If True, includes the OM (Organ Measurements) domain harmonization section.
        bw (bool): If True, includes the BW (Body Weight) domain harmonization section.
        lb (bool): If True, includes the LB (Laboratory Tests) domain harmonization section.
        mi (bool): If True, includes the MI (Microscopic Findings) domain harmonization section.

    Notes:
        - The pipeline is optimized for CDISC SEND-formatted data but can be adapted for other formats.
        - Each domain section includes both automatic and manual mapping tools, with support for collaborative review via a sharable web interface (https://onith-sharable-app.streamlit.app/).
        - The final notebook includes export functionality to save harmonized datasets and documentation.
        - Users must configure project-specific parameters (e.g., directory paths, column names) in the generated notebook before execution.

    Output:
        A Jupyter notebook named 'harmonization_pipeline.ipynb' saved in the specified directory.
    """
    
    # define notebook_path
    os.makedirs(notebook_dir, exist_ok=True)
    notebook_path = Path(notebook_dir) / 'harmonization_pipeline.ipynb'
    notebook_path = notebook_path.as_posix()

    nb = nbf.v4.new_notebook()
    cells = []

    # General introduction
    cells.append(nbf.v4.new_markdown_cell("""# Harmonization Pipeline (ontology-based non-clinical integration and term harmonization)

**What is this?:** This notebook contains a collection of functions designed to harmonize various data domains from non-clinical studies, making them suitable for cross-study integration and machine learning applications.
It is optimized for data in the SEND format (CDISC SEND standard), but it can be adapted for other data formats as needed.

**Why it matters:** When combining data across studies, the terminology and units used to describe specific findings in animals can vary significantly depending on the year, study site, and involved researchers.
Since each institution or company may have its own internal documentation system, leading to different collections of terms etc., this pipeline is designed to guide a cross-study harmonization process and its documentation, while supporting continuous customization.

**How it works:** Each domain has its own set of domain-specific functions, organized into dedicated classes. These functions are already arranged in the correct execution order within this notebook, with the export and documentation step as last step of each domain-specific pipeline section. This way, all decisions made during the harmonization process are documented to ensure reproducability.

**Getting started:** To begin the harmonization process, configure the `HarmonizerBase` in the next section by specifying the appropriate directory paths and project settings.

"""))


    # Import statements 
    #todo as soon as python package uploaded: only needed to import onith once -> in __init__ all modules will be loaded automatically
    cells.append(nbf.v4.new_code_cell("from onith import *"))

    # Configuration cell
    cells.append(nbf.v4.new_code_cell(
"""# Configure your project settings here: 
temp_dir        = "<your_temp_directory>"
output_dir      = "<your_output_directory>"
project_name    = "<your_project_name>"
sample_column   = "USUBJID" # adjust if necessary
study_id_column = "STUDYID" # adjust if necessary
group_id_column = "ARMCD" # adjust if necessary. The group id should be integer (0 for control samples, >0 for treated samples)"""))
    
    cells.append(nbf.v4.new_code_cell(
        "# Instantiate the HarmonizerBase class with your project configuration.\n"
        "# This sets up the environment and shared metadata for all domain-specific harmonization tasks.\n"
        "# You can customize column names if your dataset deviates from SEND conventions.\n"
        "project = HarmonizerBase(temp_dir=temp_dir,\n"
        "                         output_dir=output_dir,\n"
        "                         project_name=project_name,\n"
        "                         sample_column=sample_column,\n"
        "                         study_id_column=study_id_column,\n"
        "                         group_id_column=group_id_column)"))

    cells.append(nbf.v4.new_markdown_cell("""### Extracting Treatment Group Information from Sample Metadata
- **Input**: 
  - Case 1: If you are working with SEND formatted data, you can directly use the **DM (Demographics)** domain as input for the `extract_metadata` function to automatically retrieve the treatment group information
  - Case 2: If your dataset include the following columns but does'nt follow the SEND column naming convention, you have to configure the colum name parameters of the `extract_metadata` function with your specific column names.
    - sample ID column = <your_sample_column> (equivalent to "USUBJID" in the SEND column naming convention)
    - study ID column (equivalent to "STUDYID" in the SEND column naming convention)
    - An integer treatment group ID column (equivalent to "ARMCD" in the SEND column naming convention)
    - A treatment group name column (equivalent to "ARM" in the SEND column naming convention)
  - Case 3: If your sample metadata does'nt include these columns, you can manually prepare the metadata table and save it as "metadata_<your_project_name>.csv in your output folder <your_output_dir>.
    It should include a column <your_sample_column> and a column "ARMCD" with the group IDs as described below.
- **Output**: A metadata table containing:
  - **sample ID** (<your_sample_column>)
  - **group ID** (column "ARMCD")
    - `0` = Control group  
    - `>0` = Treatment group
- **Important Notes**:
  - The treatment group information is essential for continuing the harmonization process.
  - Any samples not listed in the metadata table will be excluded from the harmonization of all domains.
"""))
    
    cells.append(nbf.v4.new_code_cell(
        "# Fetch example data path (delete this cell if working with your own data)\n"
        f"example_data_path = r\"{example_data_path}\"\n"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Load and extract metadata from the demographics dataset\n"
        "metadata = pd.read_csv(os.path.join(example_data_path, \"DM.csv\"), sep=\";\")\n"
        "metadata = project.extract_metadata(metadata,\n"
        "                                    output_dir=output_dir,\n"
        "                                    project_name=project_name,\n"
        "                                    sample_column=sample_column,\n"
        "                                    include_recovery_animals=False)"))
    
    # MI domain section
    if mi:
        cells.append(nbf.v4.new_markdown_cell("""## MI Domain

The MI Domain within SEND data encompasses a structured collection of *histopathological findings*, recorded across multiple organs for each animal.

Each **finding** is composed of two key components:
- **Organ**: The anatomical site examined.
- **Lesion**: The specific pathological alteration observed in that organ, as described by the responsible pathologist.

To ensure consistent and standardized terms, this section of the harmonization pipeline includes a set of functions that map both organ and lesion terms to established **industry-standard ontologies** (HPATH for lesion terms, INHAND for organ terms). These ontologies are enriched with synonym sets, enabling input terms to be matched based on word similarity. 

Any input terms, whether organ or lesion, that cannot be confidently mapped to a corresponding synonym or primary term will be transferred to a **shareable application** featuring an intuitive user interface.

This sharable application (https://onith-sharable-app.streamlit.app/) supports interactive manual mapping of unresolved terms and is designed to facilitate interdisciplinary collaboration. It enables subject-matter experts, including those without coding expertise, to review and resolve ambiguous or novel terminology efficiently.

To start the harmonization, configure the `MIHarmonizer` with custom column names if needed and import your dataframe with the animal, organ and lesion information. 

Note: Histopathological datasets usually consist to a large extend out of "NORMAL" entries. Documenting, that the pathologist investigated the respective tissue section, but didn't find any abnormalities. For clarity, these entries are temporarily removed from the dataset while mapping the findings to the organ and lesion ontology and added back in once the harmonization is applied. The lesion term of those entries is harmonized to "NORMAL".
"""))
        cells.append(nbf.v4.new_code_cell(
            "# No configuration of additional parameters is needed if column names follow the SEND column naming convention\n"
            "# (USUBJID for animal IDs, MISPEC for organ terms, and MIORRES for lesion terms).\n"
            "# Otherwise, add your custom column names as attributes to the MIHarmonizer class.\n"
            "mi = MIHarmonizer(temp_dir=temp_dir,\n"
            "                  output_dir=output_dir,\n"
            "                  project_name=project_name,\n"
            "                  sample_column=sample_column,\n"
            "                  study_id_column=study_id_column,\n"
            "                  group_id_column=group_id_column,\n"
            "                  lesion_column = \"MISTRESC\")"))

        cells.append(nbf.v4.new_code_cell(
            "# To test the pipeline: Load the example dataset\n"
            "df_raw = pd.read_csv(os.path.join(example_data_path, \"MI.csv\"), sep=\";\")"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Perform automatic organ term mapping\n"
            "df = mi.clean_mi(df_raw) #add the parameter metadata_path if metadata is provided manually.m\n"
            "df = mi.automatic_mapping(df, input_type=\"organ\")"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# If there are terms for which manual mapping is needed: launch the manual mapping editor and follow the instructions within the application:\n"
            "# If all terms were mapped automatically, jump to the next step (unite_and_save_mappings) and set the parameter 'integrate_manual_mappings' to False.\n"
            "# If you need to delegate the manual mapping to a colleague, you can send him/her the publically available sharable app for manual mapping (https://onith-sharable-app.streamlit.app/).\n"
            "# Additionally to the link, share the input JSON file with the respective colleague (see cell output for file name and location).\n"
            "mi.launch_manual_mapping(df, input_type=\"organ\")"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Generate and save final organ mapping\n"
            "# Set 'integrate_manual_mapping' to True if the results of the manual mapping editor shall be integrated.\n"
            "# If all terms were already mapped automatically, set it to False.\n"
            "# Optional: update your local copy of the organ term ontology with the manually mapped synonyms from your dataset.\n"
            "organ_mapping = mi.unite_and_save_mappings(df,\n"
            "                                           update_dict=False,\n"
            "                                           input_type=\"organ\",\n"
            "                                           integrate_manual_mapping=True)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Perform automatic lesion term mapping\n"
            "df = mi.clean_mi(df_raw) #add the parameter metadata_path if metadata is provided manually.\n"
            "df = mi.automatic_mapping(df, input_type=\"lesion\")"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Launch manual review for lesion mappings - make at least one term selection and save the progress using the application's export button\n"
            "# If you need to delegate the manual mapping to a colleague, you can send him/her the publically available sharable app for manual mapping (https://onith-sharable-app.streamlit.app/).\n"
            "# Additionally to the link, share the input JSON file and the hpath term info JSON file with the respective colleague (see cell output for file name and location).\n"
            "mi.launch_manual_mapping(df, input_type=\"lesion\")"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Generate and save final lesion mapping (automatic and manual mapping results combined)\n"
            "# Set 'integrate_manual_mapping' to True if the results of the manual mapping editor shall be integrated.\n"
            "# If all terms were already mapped automatically, set it to False.\n"
            "# Optional: update your local copy of the lesion term ontology with the manually mapped synonyms from your dataset to gradually refine the ontology for your use case.\n"
            "lesion_mapping = mi.unite_and_save_mappings(df,\n"
            "                                            update_dict=False,\n"
            "                                            input_type=\"lesion\",\n"
            "                                            integrate_manual_mapping=True)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Apply organ and lesion mappings to the dataset and add 'NORMAL' entries back into dataset\n"
            "df = mi.clean_mi(df_raw) #add the parameter metadata_path if metadata is provided manually.\n"
            "df = mi.apply_mapping_mi(df, organ_mapping, lesion_mapping)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Exploratory data analysis\n"
            "# Optional: set parameters 'reduce_organ_panel' and 'remove_animals_with_few_organs' to True\n"
            "# in order to reduce study design bias (organ panel) and animals with many NaN.\n"
            "df = mi.explore_harmonized_df(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "#! IMPORTANT: Save the notebook file now, before continuing with the export\n"
            "# Otherwise the current state of the outputs within the notebook will not be captured correctly."
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Export the final dataset and documentation - correct notebook_path if necessary\n"
            "# If the section header of your Jupyter notebook deviates from the standard header as set by the package's pipeline configurator:\n"
            "# configure it via custom_section_header\n"
            f"export_data_and_documentation(df=df,\n"
            f"                              output_dir=output_dir,\n"
            f"                              notebook_path=\"{notebook_path}\",\n"
            f"                              domain=\"mi\",\n"
            f"                              project_name=project_name)"
        ))


    # LB domain section
    if lb:
        cells.append(nbf.v4.new_markdown_cell(
            "## LB Domain\n\n"
            "The **LB Domain** in SEND data refers to laboratory test results (referred to as *markers*) collected from biological specimens "
            "(e.g., blood, urine) of animals during a study to evaluate their physiological and pathological status.\n\n"
            "Each **marker** has several key attributes:\n"
            "- **Marker term / name**: The term used to describe the test and analysis performed (e.g., 'BASO', 'BASOa', 'Basophils', 'Basophil Counts' all referring to the same measurement).\n"
            "- **Marker unit**: The unit used to measure the marker (can vary based on lab, machine, or company).\n"
            "- **Specimen**: The material the marker was measured in (typically urine, whole blood, plasma, etc.).\n"
            "- **Time point**: Depending on the study design and specimen, a marker can be measured once (e.g., at necropsy) or multiple times during the study.\n\n"
            "This harmonization pipeline ensures:\n"
            "- Consistency in marker terms and units (based on SEND CDISC Terminology), using both automatic and manual mapping.\n"
            "- A *sharable and interactive application* for interdisciplinary collaboration on unresolved mappings: https://onith-sharable-app.streamlit.app/\n"
            "- Interactive selection of a suitable marker panel (i.e., markers measured consistently across studies).\n"
            "- Identification and handling of non-numeric values (e.g., 'below linearity').\n"
            "- Harmonization of multiple timepoint measurements.\n"
            "- Statistical normalization.\n"
            "- Outlier detection and imputation."
            "To start the harmonization, configure the `MIHarmonizer` with custom column names if needed. Necessary columns for the LB harmonization are: study id, sample id, marker term, marker value, marker unit, measurement day, specimen"
        ))


        # Code cells
        cells.append(nbf.v4.new_code_cell(
            "# No configuration of additional parameters is needed if column names follow the SEND column naming convention\n"
            "# (USUBJID for animal IDs, LBSPEC for specimen terms, STUDYID for study ID, LBSTRESU for units, LBDY for measurement day, LBTESTCD for marker name).\n"
            "# Otherwise, add your custom column names as attributes to the LBHarmonizer class.\n"
            "lb = LBHarmonizer(temp_dir=temp_dir,\n"
            "                  output_dir=output_dir,\n"
            "                  project_name=project_name,\n"
            "                  sample_column=sample_column,\n"
            "                  study_id_column=study_id_column,\n"
            "                  group_id_column=group_id_column)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Load the raw LB dataset\n"
            "df = pd.read_csv(os.path.join(example_data_path, \"LB.csv\"), low_memory=False, sep=\";\") # to test the pipeline, load the example dataset"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Automatically map marker terms to the standard term name as suggested by the SEND CDISC Terminology\n"
            "df = lb.automatic_mapping_lb(df)"
        ))

        cells.append(nbf.v4.new_markdown_cell(
            "If there are terms for which manual mapping is needed: launch the manual mapping editor and follow the instructions within the application:\n"
            "If all terms were mapped automatically, jump to the next step (unite_and_save_mappings) and set the parameter 'integrate_manual_mappings' to False.\n"
            "If you need to delegate the manual mapping to a colleague, you can send him/her the publically available sharable app for manual mapping (https://onith-sharable-app.streamlit.app/).\n"
            "Additionally to the link, share the input JSON file with the respective colleague (see cell output for file name and location).\n"))
        
        cells.append(nbf.v4.new_code_cell("lb.launch_manual_mapping_lb(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Apply the mapping to the dataset for harmonization\n"
            "# configure parameters based on function description.\n"
            "df = lb.apply_mapping_lb(df,\n"
            "                      integrate_manual_mapping=False,\n"
            "                      delete_unmapped=False,\n"
            "                      update_dict=False,\n"
            "                      add_specimen=True)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Interactively define your marker panel based on relative frequency within the sample group\n"
            "select_marker_panel = lb.define_marker_panel(df, rel_freq_group=\"sample\")"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Apply the selected marker panel to filter the dataset for the selected markers\n"
            "df = lb.apply_marker_panel(df, select_marker_panel)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Explore and handle non-numeric values in the dataset\n"
            "lb.explore_nonnumeric(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Harmonize units across the dataset to ensure consistency before calculating relative statistics\n"
            "df = lb.explore_and_harmonize_units(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Calculate control group statistics and extract treated group data\n"
            "control_stats_df, treated_df = lb.control_stats(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Reduce the number of timepoints to one per animal and marker (selected based on most significant change respect the respective control group)\n"
            "df = lb.reduce_timepoints(control_stats_df, treated_df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Normalize the data using z-score transformation\n"
            "df = lb.calculate_zscore(df, 0.01)"
            "# Pivot the dataset for analysis\n"
            "df = lb.pivot_and_sortout(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Explore the distribution of values across tests and groups\n"
            "lb.explore_distribution(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Remove outliers and impute missing values (KNN imputation)\n"
            "df = lb.outlier_removal_and_imputation(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "#! IMPORTANT: Save the notebook file now, before continuing with the export\n"
            "# Otherwise the current state of the outputs within the notebook will not be captured correctly."
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Export the final dataset and documentation - correct notebook_path if necessary\n"
            "# If the section header of your Jupyter notebook deviates from the standard header as set by the package's pipeline configurator:\n"
            "# configure it via custom_section_header\n"
            f"export_data_and_documentation(df=df,\n"
            f"                              output_dir=output_dir,\n"
            f"                              notebook_path=\"{notebook_path}\",\n"
            f"                              domain=\"lb\",\n"
            f"                              project_name=project_name)"
        ))


    # OM domain section
    if om:
        cells.append(nbf.v4.new_markdown_cell("""## OM Domain

The **OM Domain** in SEND data refers to organ measurements collected from animals at the end of the study during necropsy to get first hinds towards drug-induced alterations.

Each **measurement** has several key attributes:
- **Organ term / name**: Which organ was being measured (e.g., 'Heart', 'Liver', 'Kidney').
- **Measurement term**: The type of parameter used to report the measurement (e.g., 'Weight', 'Organ to Body Weight Ratio').
- **Measurement unit**: The unit used to measure the organ (can vary based on lab, machine, or company).
- **Measurement value**: The actual value obtained from the measurement.

This harmonization pipeline ensures:
- Consistency in organ terms, using both automatic and manual mapping, providing a *sharable and interactive application* (https://onith-sharable-app.streamlit.app/) for interdisciplinary collaboration for manual mapping of unresolved organ mappings.
- Identification and handling of non-numeric values.
- Statistical normalization.
- Outlier detection and imputation.
- Exploration of the distribution.

To start the harmonization, configure the `OMHarmonizer` with custom column names if needed. Necessary columns for the OM harmonization are: study id, animal id, organ term, measurement term, measurement value."""))
                            
        # Code cells
        cells.append(nbf.v4.new_code_cell(
            "# No configuration of additional parameters is needed if column names follow the SEND column naming convention\n"
            "# (USUBJID for animal IDs, OMSPEC for organs, STUDYID for study ID, OMSTRESU for units, OMTEST for parameter type, OMSTRESC for value).\n"
            "# Otherwise, add your custom column names as attributes to the LBHarmonizer class.\n"
            "om = OMHarmonizer(temp_dir=temp_dir,\n"
            "                  output_dir=output_dir,\n"
            "                  project_name=project_name,\n"
            "                  sample_column=sample_column,\n"
            "                  study_id_column=study_id_column,\n"
            "                  group_id_column=group_id_column)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Load the raw OM dataset\n"
            "df = pd.read_csv(os.path.join(example_data_path, \"OM.csv\"), sep=\";\") # to test the pipeline, load the example dataset"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Clean and reduce the OM dataframe\n"
            "df = om.clean_om(df) #add the parameter metadata_path if metadata is provided manually."
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Automatically map organ terms (OMSPEC) to standard terms (based on INHAND)\n"
            "organ_mapping = om.automatic_mapping(df, \"organ\")"
        ))

        cells.append(nbf.v4.new_markdown_cell(
            "If there are terms for which manual mapping is needed: launch the manual mapping editor and follow the instructions within the application:\n"
            "If all terms were mapped automatically, jump to the next step (unite_and_save_mappings) and set the parameter 'integrate_manual_mappings' to False.\n"
            "If you need to delegate the manual mapping to a colleague, you can send him/her the publically available sharable app for manual mapping (https://onith-sharable-app.streamlit.app/).\n"
            "Additionally to the link, share the input JSON file with the respective colleague (see cell output for file name and location)."
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Skip, if no manual mapping is needed\n"
            "om.launch_manual_mapping(organ_mapping, input_type=\"organ\")"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Generate and save final organ mapping. Don't skip, even if no manual mapping was performed\n"
            "organ_mapping = om.unite_and_save_mappings(organ_mapping,\n"
            "                                           update_dict=False,\n"
            "                                           input_type=\"organ\",\n"
            "                                           integrate_manual_mapping=False)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Apply organ mapping to harmonize the dataset\n"
            "df = om.apply_mapping_om(df, organ_mapping)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Explore and handle non-numeric values in the dataset\n"
            "om.explore_nonnumeric(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Explore what parameter types are covered best in the dataset (e.g. 'Weight' vs. 'Organ to Body Weigth Ratio')\n"
            "om.explore_parameter_frequency(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Prepare OM data for calculating control group statistics\n"
            "df = om.prepare_om_for_control_stats(df, parameter_type=\"Weight\")"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Calculate control group statistics and extract treated group data\n"
            "control_stats_df, treated_df = om.control_stats(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Normalize the data using z-score transformation\n"
            "df = om.add_control_mean(control_stats_df, treated_df)\n"
            "df = om.calculate_zscore(df, 0.01)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Pivot the dataset\n"
            "df = om.pivot_om(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Explore the distribution of organ weights\n"
            "om.explore_distribution(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Remove outliers and impute missing values (KNN imputation)\n"
            "df = om.outlier_removal_and_imputation(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "#! IMPORTANT: Save the notebook file now, before continuing with the export\n"
            "# Otherwise the current state of the outputs within the notebook will not be captured correctly."
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Export the final dataset and documentation - correct notebook_path if necessary\n"
            "# If the section header of your Jupyter notebook deviates from the standard header as set by the package's pipeline configurator:\n"
            "# configure it via custom_section_header\n"
            f"export_data_and_documentation(df=df,\n"
            f"                              output_dir=output_dir,\n"
            f"                              notebook_path=\"{notebook_path}\",\n"
            f"                              domain=\"om\",\n"
            f"                              project_name=project_name)"
        ))


    # BW domain section
    if bw:
        cells.append(nbf.v4.new_markdown_cell("""## BW Domain

The **BW Domain** in SEND data refers to body weight measurements collected from animals throughout the study. 

Each **body weight measurement** includes:
- **Study ID**: Identifier for the study.
- **Sample ID**: Unique identifier for each animal/sample.
- **Parameter type**: Usually either "Body Weight" or "Terminal Body Weight".
- **Body Weight Value**: The measured value.
- **Unit**: The unit the weight was measured in.

This harmonization pipeline ensures:
- Cleaning and preparation of the raw data.
- Exploration of parameter frequency.
- Filtering for terminal body weight.
- Calculation of control group statistics.
- Z-score normalization.
- Pivoting the dataset for analysis.
- Distribution exploration.
- Outlier removal and imputation.
- Export of the final dataset and documentation.
"""))

        # Code cells
        cells.append(nbf.v4.new_code_cell(
            "# No configuration of additional parameters is needed if column names follow the SEND column naming convention\n"
            "# (USUBJID for animal IDs, STUDYID for study ID, BWSTRESU for units, BWTEST for parameter type, BWSTRESC for value).\n"
            "# Otherwise, add your custom column names as attributes to the LBHarmonizer class.\n"
            "bw = BWHarmonizer(temp_dir=temp_dir,\n"
            "                  output_dir=output_dir,\n"
            "                  project_name=project_name,\n"
            "                  sample_column=sample_column,\n"
            "                  study_id_column=study_id_column,\n"
            "                  group_id_column=group_id_column)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Load the raw BW dataset\n"
            "df = pd.read_csv(os.path.join(example_data_path, \"BW.csv\"), sep=\";\") # load the example dataset to test out the pipeline"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Clean and reduce the BW dataframe\n"
            "df = bw.clean_bw(df) #add the parameter metadata_path if metadata is provided manually."
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Explore the parameter type coverage in the dataset\n"
            "bw.explore_parameter_frequency(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Filter for terminal body weight values\n"
            "df = bw.filter_for_terminal_weight(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Calculate control group statistics and extract treated group data\n"
            "control_stats_df, treated_df = bw.control_stats(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Normalize the data using z-score transformation\n"
            "df = bw.add_control_mean(control_stats_df, treated_df)\n"
            "df = bw.calculate_zscore(df, 0.01)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Pivot the dataset\n"
            "df = bw.pivot_bw(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Explore the distribution of body weights\n"
            "bw.explore_distribution(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Remove outliers and impute missing values (KNN imputation)\n"
            "df = bw.outlier_removal_and_imputation(df)"
        ))

        cells.append(nbf.v4.new_code_cell(
            "#! IMPORTANT: Save the notebook file now, before continuing with the export\n"
            "# Otherwise the current state of the outputs within the notebook will not be captured correctly."
        ))

        cells.append(nbf.v4.new_code_cell(
            "# Export the final dataset and documentation\n"
            "export_data_and_documentation(df=df,\n"
            "                              output_dir=output_dir,\n"
           f"                              notebook_path=\"{notebook_path}\",\n"
            "                              domain=\"bw\",\n"
            "                              project_name=project_name)"
        ))


    # generate and save notebook 
    nb['cells'] = cells

    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

    print(f"The notebook has been created successfully and saved as 'harmonization_pipeline.ipynb' in {notebook_dir}.")
