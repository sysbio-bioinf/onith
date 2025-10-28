# Onith

## Links
- Paper: (link)
- PyPI: (link)
- GitHub: (link)

## Installation

ONITH is available on PyPI and can be installed with pip:

for alpha tester:
cd into the package folder (the one containing the pyproject.toml)
```bash
pip install .
```

for final users:
```bash
pip install onith
```

## Concept
**What is this?** This package is a collection of classes and functions designed to harmonize various data domains from non-clinical studies, making them suitable for cross-study integration and machine learning applications.

It is optimized for data in the SEND format (CDISC SEND standard), but it can be adapted for other data formats as needed.

**Why it matters** When combining data across studies, the terminology and units used to describe specific findings in animals can vary significantly depending on the year, study site, and involved researchers.

Since each institution or company may have its own internal documentation system, leading to different collections of terms etc., this pipeline is designed to guide a cross-study harmonization process and its documentation, while supporting continuous customization.

**How it works** Each domain has its own set of domain-specific functions, organized into dedicated classes. These functions are already arranged in the correct execution order within this notebook, with the export and documentation step as last step of the pipeline. This way, all decisions made during the harmonization process are documented to ensure reproducability.


## Getting started

Generate your first customized harmonization pipeline:

```python
from onith import *

configure_harmonization_pipeline("<your_output_directory>", lb = True, mi = True, bw = True, om = True)
```

The function ```configure_harmonization_pipeline``` will generate a **custom jupyter notebook file** in the specified output directory.

To configure the notebook, you have to specify what data domains you want to harmonize.

For each specified domain, the custom harmonization pipeline will include a section with all necessary domain-specfic functions already arranged in the correct execution order to ensure a unbiased and reproducable human-in-the-loop harmonization process.

- ```lb```: If set to True, the custom harmonization pipeline will include a section dedicated to the harmonization of the **LB data domain (Laboratory Test Results = Blood Marker Data)**
- ```mi```: If set to True, the custom harmonization pipeline will include a section dedicated to the harmonization of the **MI data domain (Microscopic Findings = Histopathological Finding Descriptions)**
- ```bw```: If set to True, the custom harmonization pipeline will include a section dedicated to the harmonization of the **BW data domain (Body Weight)**
- ```om```: If set to True, the custom harmonization pipeline will include a section dedicated to the harmonization of the **OM data domain (Organ Measurements)**

The generated harmonization pipeline includes default paths to example data, allowing you to explore and test the process before loading your own datasets.

## Package Architecture
![Package Architecture](Package_Architecture.png)

## Documentation

Step-by-step instructions for all following steps will be given in the generated custom notebook file, based on the harmonization pipeline configuration.
Function-specific documentation is available on PyPI (link).

## License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with proper attribution. See the [LICENSE](LICENSE) file for more details.

## Citation

Please cite our work in your publications if this package contributed to your project.
