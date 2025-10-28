import pandas as pd
from typing import Literal
import pandas as pd
import re
from pathlib import Path

def load_normal_entries(as_dict: bool = True):
    """
    Return known 'normal' lesion terms either as a dictionary or a list.

    Parameters:
        as_dict (bool): If True, returns a dictionary mapping each term to 'NORMAL'.
                        If False, returns a list of terms.

    Returns:
        dict or list: Dictionary or list of normal lesion terms.
    """
    
    normal_terms = [
        "NORMAL",
        "NVL",
        "UNREMARKABLE",
        "NO CORRELATING LESION",
        "NO CHANGES OBSERVED",
        "NO CHANGES OBSERVER",
        "NO SECTION",
        "ARTIFACT",
        "ANIMAL ID",
        "ARTIFACT(S)",
        "INCLUSIONS, HEPATOCYTE, ARTIFACT",
        "NOS",
        "PARS NERVOSA NOT CUT IN SECTION",
        "UNDETERMINED STAGE OF ESTRUS CYCLE",
        "NAN"
    ]

    if as_dict:
        return {term: "NORMAL" for term in normal_terms}
    else:
        return normal_terms
    

def remove_normal_mi_entries(df, column_name, normal_entry_keywords: list = None):
    """Remove rows where the specified column contains known 'normal' lesion terms."""
    
    if normal_entry_keywords is None:
        normal_entry_keywords = load_normal_entries()

    df = df[~df[column_name].isin(normal_entry_keywords)].dropna(subset=[column_name])
    
    return df


def load_unit_dict() -> dict:
    """
    Return a dictionary mapping various unit notations to standardized forms.

    Returns:
        dict: Dictionary of unit mappings.
    """

    unit_dict = {
        "STRESU": "Unit",
        "10**3/ul": "10^3/ul",
        "10**6/ul": "10^6/ul",
        "10^9/l": "10^3/ul",
        "10^12/l": "10^6/ul",
        "vol%": "%",
        "%(v/v)": "%",
        "l/l": "%"
    }
    
    return unit_dict


def strip_last_segment(term):    
    """Removes the last comma-separated segment from a term if more than two segments are present.
    Used during automatic term mapping of lesion terms for stepwise removal of granularity in the lesion description."""
    
    segments = term.split(',')
    if len(segments) > 2:
        return ','.join(segments[:-1])
    return term


def harmonize_formatting(
    df: pd.DataFrame, 
    columns_to_harmonize: list[str]
) -> pd.DataFrame:
    """
    Harmonizes and standardizes the text formatting of lesion and organ terms of both, ontology and input dataset, to facilitate mapping.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the columns to be harmonized (mostly either the ontology or input data)

    columns_to_harmonize : list of str
        A list of column names in the DataFrame that should be harmonized. Each column should contain string or
        string-convertible values.

    Returns:
    -------
    pandas.DataFrame
        The modified DataFrame with harmonized text formatting in the specified columns. Duplicate rows are removed.
    """
    
    df = df.copy()  # Avoid modifying the original DataFrame

    for column in columns_to_harmonize:
        if column in df.columns:
            df.loc[:, column] = df[column].astype(str).str.upper()
            df.loc[:, column] = df[column].str.replace('.', '', regex=False)
            df.loc[:, column] = df[column].str.replace(' and ', ',', regex=False)
            df.loc[:, column] = df[column].str.replace(' or ', ',', regex=False)
            df.loc[:, column] = df[column].str.replace(' / ', ',', regex=False)
            df.loc[:, column] = df[column].str.replace('/ ', ',', regex=False)
            df.loc[:, column] = df[column].str.replace('/S', 'S', regex=False)
            df.loc[:, column] = df[column].str.replace('/', ',', regex=False)
            df.loc[:, column] = df[column].str.replace(' - ', ',', regex=False)
            df.loc[:, column] = df[column].str.replace('-', ',', regex=False)
            df.loc[:, column] = df[column].str.replace('; ', ',', regex=False)
            df.loc[:, column] = df[column].str.replace(';', ',', regex=False)
            df.loc[:, column] = df[column].str.replace(', ', ',', regex=False)
            df.loc[:, column] = df[column].str.replace(': ', ',', regex=False)
            df.loc[:, column] = df[column].str.replace(' :', ',', regex=False)
            df.loc[:, column] = df[column].str.replace(':', ',', regex=False)
            df.loc[:, column] = df[column].str.replace(',', ', ', regex=False)
            df.loc[:, column] = df[column].apply(lambda x: re.sub(r'\(.*?\)|\[.*?\]', '', x))

    df = df.replace("NAN", None)
    df = df.drop_duplicates()

    return df
    

def parse_obo_content(file_path):
    """
    Parses an OBO (Open Biomedical Ontologies) file and extracts structured term data.

    Parameters:
    -----------
    file_path : str
        Path to the OBO file to be parsed.

    Returns:
    --------
    List[dict]
        A list of dictionaries, each representing a term in the ontology.
        Each dictionary contains key-value pairs for fields such as:
        - 'id'
        - 'name'
        - 'def'
        - 'synonym' (as a list if multiple)
        - 'xref', 'is_a', 'alt_id', 'subset' (as lists if present)

    Notes:
    ------
    - The function preserves multiple values for keys like 'synonym', 'xref', 'is_a', etc., as lists.
    - Quotation marks around values are stripped.
    - Empty lines between terms are used to separate term blocks.
    - The '[Term]' header is ignored during parsing.
    """
    
    with open(file_path, 'r') as file:
        content = file.read()

    terms = content.strip().split("\n\n")
    data = []
    for term in terms:
        term_data = {}
        lines = term.split("\n")
        for line in lines:
            if line.startswith("[Term]"):
                continue
            if ": " in line:
                key, value = line.split(": ", 1)
                value = value.strip().strip('"')
                if key in term_data:
                    # If the key already exists, append to the list
                    if isinstance(term_data[key], list):
                        term_data[key].append(value)
                    else:
                        term_data[key] = [term_data[key], value]
                else:
                    # For keys that can appear multiple times, store as list
                    if key in {"synonym", "xref", "is_a", "alt_id", "subset"}:
                        term_data[key] = [value]
                    else:
                        term_data[key] = value
        data.append(term_data)
    return data


def convert_obo_to_df(file_path: str, input_type : Literal["organ","lesion", "lb"]) -> pd.DataFrame:
    """
    Converts an OBO ontology file into a structured pandas DataFrame, with optional parsing of relationships
    and filtering based on the input type.

    Parameters:
    -----------
    file_path : str
        Path to the OBO file to be parsed.
    input_type : Literal["organ", "lesion"]
        Specifies which ontology being parsed. Determines how the data is filtered and which columns are processed.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing parsed ontology terms with relevant metadata. Includes:
        - 'id', 'name', 'def', and optionally 'synonym'
        - 'parent_id' and 'parent_name' (from 'is_a' relationships)
        - For lesions: 'relationship_type', 'relationship_id', 'relationship_name'

    Notes:
    ------
    - Uses `parse_obo_content()` to extract raw term data.
    - Explodes multi-valued fields like 'is_a' and 'relationship' into separate rows.
    """
            
    data = parse_obo_content(file_path)
    df = pd.DataFrame(data)

    # Split up parent and relationship information
    if input_type in ["lesion","organ"]:
        df = df.explode("is_a")
        if 'is_a' in df.columns:
            df[["parent_id", "parent_name"]] = df["is_a"].str.split(" ! ", expand=True)
                
    if input_type == "lesion":

        df = df.explode("relationship")
        if 'relationship' in df.columns:
            df[["relationship_id", "relationship_name"]] = df["relationship"].str.split(" ! ", expand=True)
            df[["relationship_type", "relationship_id"]] = df["relationship_id"].str.split(" ", expand=True)
            
        # Remove obsolete entries
        if 'is_obsolete' in df.columns:
            df = df[df["is_obsolete"].isna()]
            
        # Drop unnecessary columns and rows
        columns_to_drop = ['data-version', 'xref', 'last manual completion with additional synonyms', 'format-version', 'date', 'auto-generated-by', 'relationship', 'ontology', "is_a", "is_obsolete", "replaced_by", "comment", "is_transitive"]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        df['id'] = df['id'].astype(str)
        df = df[df['id'].str.startswith('MC')]
        
    if input_type in ["organ", "lb"]:
        columns_to_drop = ['date', 'last manual completion with additional synonyms']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        df = df.dropna(subset="name")

    return df


def transform_synonyms(df):
    """
    Transforms the ontology (after being converted to a dataframe) by exploding the 'synonym' column into individual rows.
    """

    # Explode synonym list to individual rows
    df_syn = df.explode('synonym').reset_index(drop=True)
    
    # Separate synonym and synonym type into individual columns
    df_syn['synonym'] = df_syn['synonym'].str.split('"').str[0].str.strip()
    
    # Return only the synonym column
    return df_syn


def load_and_prepare_dict(input_type : Literal["organ","lesion", "lb"]) -> pd.DataFrame:
    """
    Loads and prepares the lesion or organ ontology by importing the most recent version from the packages' dict folder,
    transforming it to access the individual synonym, and harmonizing terms.

    Returns
    -------
    pd.DataFrame :
        The prepared ontology DataFrame.
    """
    
    base_path = Path(__file__).resolve().parent / "dict"

    if input_type == "lesion":
        dict_path = base_path / "hpath_ontology.obo"
        columns_to_harmonize = ["name", "synonym", "parent_name", "relationship_name"]
    elif input_type == "organ":
        dict_path = base_path / "organ_ontology.obo"
        columns_to_harmonize = ["name", "synonym", "parent_name"]
    elif input_type == "lb":
        dict_path = base_path / "lb_terminology.obo"
        columns_to_harmonize = ["name", "synonym"]
    else:
        raise ValueError("Invalid input_type. Choose from 'organ', 'lesion', or 'lb'.")

    dict = convert_obo_to_df(str(dict_path), input_type)    
    dict = transform_synonyms(dict)
    dict = harmonize_formatting(df=dict, columns_to_harmonize=columns_to_harmonize)
    
    # add original term as new row to synonym column and remove duplicates 
    main_terms = dict['name'].unique()
    # Find all (name, synonym) pairs that already exist
    existing_pairs = set(zip(dict['name'], dict['synonym']))
    # List to collect new rows
    new_rows = []
    
    for name in main_terms:
        # If (name, name) is not already present, add a new row
        if (name, name) not in existing_pairs:
            # Get the first row for this name (to copy other columns)
            row = dict[dict['name'] == name].iloc[0].copy()
            row['synonym'] = name
            new_rows.append(row)
    
    # If there are new rows, append them to the DataFrame
    if new_rows:
        dict = pd.concat([dict, pd.DataFrame(new_rows)], ignore_index=True)

    dict = dict.drop_duplicates()
    dict["synonym"] = dict["synonym"].str.rstrip(",")
    dict["synonym"] = dict["synonym"].str.rstrip("-")
    
    dict['synonym'] = dict.apply(lambda row: row['name'] if row['synonym'] == '' else row['synonym'], axis=1)
    
    dict = dict.dropna(subset="name")
    
    return dict


def hpath_syn_to_main_dict(input_type):
    """
    As preparation for the info card with term information in the manual mapping editor: 
    Generates synonym-to-main term mappings from the HPath ontology.
    """

    hpath_syn = load_and_prepare_dict(input_type)
    hpath_syn = harmonize_formatting(hpath_syn, ["name", "synonym"])
    hpath_syn = hpath_syn[["name", "synonym"]].dropna()
    hpath_syn_dict = hpath_syn.set_index("synonym")["name"].to_dict()
    
    return hpath_syn_dict


def hpath_term_info_to_json(input_type):
    """
    As preparation for the info card with term information in the manual mapping editor: 
    Generates a JSON-serializable dictionary containing main term metadata and synonym-to-main term mappings from the HPath ontology.
    """

    term_info = {}
    synonym_to_main = {}

    hpath_obo_path = Path(__file__).resolve().parent / "dict/hpath_ontology.obo"

    hpath = convert_obo_to_df(hpath_obo_path, input_type)
    hpath = harmonize_formatting(hpath, ["name", "synonym"])
    hpath_syn_dict = hpath_syn_to_main_dict(input_type)

    for _, row in hpath.iterrows():
        main_term = row['name']
        definition = str(row['def']).strip('" []').strip('"')
        parent_term = row['parent_name']

        term_info[main_term] = {
            "main_term": main_term,
            "parent_term": parent_term,
            "definition": definition
        }

    for synonym, main_term in hpath_syn_dict.items():
        synonym_to_main[synonym] = main_term

    return {
        "term_info": term_info,
        "synonym_to_main": synonym_to_main
    }