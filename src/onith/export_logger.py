import os
from datetime import datetime
import nbformat
import pandas as pd
import shutil
import nbformat
from nbconvert import HTMLExporter
import subprocess
from typing import Literal
import platform
import toml
import sys
from pathlib import Path

def save_as_obo(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves a ontology in dataframe format as an OBO file.

    Parameters:
    df (pd.DataFrame): The ontology in dataframe format that has to be saved (e.g. because new synonyms were added in the harmonization process).
    file_path (str): The path where the OBO file will be saved.

    Returns:
    None
    """

    with open(file_path, 'w') as f:
        for _, row in df.iterrows():
            f.write("[Term]\n")
            for col in df.columns:
                value = row[col]
                if isinstance(value, list):
                    # For lists, check if the list is empty or all elements are NA
                    if value and any(pd.notna(item) for item in value):
                        f.write(f"{col}: {value}\n")
                else:
                    # For scalars (strings, numbers), check directly
                    if pd.notna(value):
                        f.write(f"{col}: {value}\n")
            f.write("\n")
            
            
            
def save_updated_dict(obo_file_path, updated_dict, input_type:Literal["organ","lesion", "lb"]) -> None:
    """
    Saves the updated ontology DataFrame (e.g. after adding new synonyms) as an OBO file and archives the previous version with a timestamp.

    Parameters:
    df (pd.DataFrame): The updated ontology formatted dataFrame.

    Returns:
    None
    """
    base_path = Path(__file__).resolve().parent / "dict/Archive"
    
    # generate subfolder in Archive with timestamp
    name_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = os.path.join(base_path, name_timestamp)
    
    os.makedirs(subfolder, exist_ok=True)

    # move previous version of hpath_ontology.obo into its newly generated subfolder for archiving
    if input_type == "lesion":
        dict = "hpath_ontology.obo"
    if input_type == "organ":
        dict = "organ_ontology.obo"
    if input_type == "lb":
        dict = "lb_terminology.obo"
        
    shutil.move(obo_file_path, os.path.join(subfolder, dict))

    # Save the updated hpath as obo
    with open(obo_file_path, "w", encoding="utf-8") as file:
        file.writelines(updated_dict)
    
    return subfolder


# main function for export of data, notebook screenshot as html, and project metadata simultaneosly
def export_data_and_documentation(df : pd.DataFrame,  
                                  output_dir : str,  
                                  notebook_path : str,
                                  domain : Literal["mi", "lb", "om", "bw"], 
                                  project_name : str,
                                  custom_section_header : bool = False,                                 
                                  save_dict_snapshot : bool = True, 
                                  organ_dict_path = None, 
                                  lesion_dict_path = None,
                                  lb_dict_path = None,):

    """
    This function performs the following tasks:
    1. Saves the provided DataFrame to a timestamped subfolder in the specified output directory.
    2. Extracts the domain-specific section of the Jupyter notebook (based on domain or custom header) and saves it as an HTML file.
    3. Collects and saves metadata including Python version, installed packages, and pyproject.toml content.
    4. Optionally saves snapshots of organ and lesion dictionaries for reproducibility (only for the "mi" domain).

    Parameters:
    ----------
    df : pd.DataFrame
        The harmonized DataFrame to export.
    output_dir : str
        Directory where the output subfolder will be created.
    notebook_path : str
        Path to the Jupyter notebook file to extract documentation from.
    domain : Literal["mi", "lb", "om", "bw"]
        Domain identifier used to determine the notebook section and naming conventions.
    project_name : str
        Name of the project, used in filenames and metadata.
    custom_section_header : bool, optional
        Custom markdown header to identify the notebook section to export. If False, uses default domain headers as set by the pipeline configurator.
    save_dict_snapshot : bool, optional
        Whether to save snapshots of the domain-specific dictionaries.
    organ_dict_path : str, optional
        Path to the organ ontology file. Only has to be filled if the path / file name deviates from the default. 
    lesion_dict_path : str, optional
        Path to the lesion ontology file. Only has to be filled if the path / file name deviates from the default.
    lb_dict_path : str, optional
        Path to the lb terminology file. Only has to be filled if the path / file name deviates from the default.

    Returns:
    -------
    None
        Outputs are saved to disk. A message is printed indicating the export location.
    """

    base_path = Path(__file__).resolve().parent / "dict"

    organ_dict_path = organ_dict_path or base_path / "organ_ontology.obo"
    lesion_dict_path = lesion_dict_path or base_path / "hpath_ontology.obo"
    lb_dict_path = lb_dict_path or base_path / "lb_terminology.obo"

    # Generate a subfolder with timestamp
    name_timestamp = datetime.now().strftime(f"{project_name}_{domain}_output_%Y%m%d_%H%M%S")
    subfolder = os.path.join(output_dir, name_timestamp)
    os.makedirs(subfolder, exist_ok=True)

    # --- Save the harmonized df to csv ---
    file_path = os.path.join(subfolder, f"{project_name}_{domain}_harmonization_output.csv")
    df.to_csv(file_path)  

    # --- Save the current notebook section as html ---  
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
        
        # Filter cells by section header if provided  
    if custom_section_header:
        section_header = custom_section_header
    else:
        domain_headers = {
            "mi": "MI Domain",
            "lb": "LB Domain",
            "bw": "BW Domain",
            "om": "OM Domain"
        }
        section_header = domain_headers[domain]
    
    cells_to_export = []
    capture = section_header is None
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and section_header and section_header in cell.source:
            capture = True
        if capture:
            cells_to_export.append(cell)
            
        # Create a new notebook object in memory with only the domain's section 
    new_nb = nbformat.v4.new_notebook(cells=cells_to_export)

        # Convert notebook section to HTML
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'classic'
    body, _ = html_exporter.from_notebook_node(new_nb)
    with open(os.path.join(subfolder, f'{project_name}_{domain}_notebook_snapshot.html'), 'w') as f:
        f.write(body)

    # --- Save metadata ---
        # fetch information 
    timestamp = datetime.now().isoformat()
    python_version = platform.python_version()
    pip_freeze = subprocess.check_output(['pip', 'freeze']).decode('utf-8')
    pyproject_metadata = ""
    
    current_file = Path(__file__)
    pyproject_path = current_file.parents[2] / "pyproject.toml"

    if os.path.exists(pyproject_path):
        try:
            pyproject = toml.load(pyproject_path)
            pyproject_metadata = toml.dumps(pyproject.get("tool", {}).get("poetry", {}))
        except Exception as e:
            pyproject_metadata = f"Failed to parse pyproject.toml: {e}"

    metadata_path = os.path.join(subfolder, f'{project_name}_{domain}_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Project Name: {project_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Python Version: {python_version}\n\n")
        #f.write("Project Metadata (from pyproject.toml):\n")
        f.write(pyproject_metadata + "\n")
        f.write("Installed Packages (pip freeze):\n")
        f.write(pip_freeze)

    # Save current state of custom dictionaries for later reproducability
    if domain == "mi" and save_dict_snapshot:
        dicts_folder = os.path.join(subfolder, "dict_snapshots")
        os.makedirs(dicts_folder, exist_ok=True)

        # Copy organ dictionary
        if os.path.exists(organ_dict_path):
            organ_dict_dest = os.path.join(dicts_folder, os.path.basename(organ_dict_path))
            with open(organ_dict_path, 'rb') as src, open(organ_dict_dest, 'wb') as dst:
                dst.write(src.read())

        # Copy lesion dictionary
        if os.path.exists(lesion_dict_path):
            lesion_dict_dest = os.path.join(dicts_folder, os.path.basename(lesion_dict_path))
            with open(lesion_dict_path, 'rb') as src, open(lesion_dict_dest, 'wb') as dst:
                dst.write(src.read())

    if domain == "lb" and save_dict_snapshot:
        dicts_folder = os.path.join(subfolder, "dict_snapshots")
        os.makedirs(dicts_folder, exist_ok=True)
        
        # Copy lb dictionary
        if os.path.exists(lb_dict_path):
            lesion_dict_dest = os.path.join(dicts_folder, os.path.basename(lb_dict_path))
            with open(lb_dict_path, 'rb') as src, open(lesion_dict_dest, 'wb') as dst:
                dst.write(src.read())
    
    if domain == "om" and save_dict_snapshot:
        dicts_folder = os.path.join(subfolder, "dict_snapshots")
        os.makedirs(dicts_folder, exist_ok=True)

        # Copy organ dictionary
        if os.path.exists(organ_dict_path):
            organ_dict_dest = os.path.join(dicts_folder, os.path.basename(organ_dict_path))
            with open(organ_dict_path, 'rb') as src, open(organ_dict_dest, 'wb') as dst:
                dst.write(src.read())
        
    if save_dict_snapshot:    
        print(f"Harmonized dataframe, snapshot of the current notebook section as well as the state of organ and lesion dictionary and metadata exported to {output_dir}")
    else: 
        print(f"Harmonized dataframe, snapshot of the current notebook section and metadata exported to {output_dir}")
    
    return

