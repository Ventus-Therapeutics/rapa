# 1. Overview
RAPA (Rotamer And Protonation Assignment) is a command line tool that aids in protein preparation by assigning various protonation and rotamer states of unassigned amino-acids of the given structure and outputs all the models that are energetically accessible. Details of the algorithm can be found in the following manuscript:
https://chemrxiv.org/engage/chemrxiv/article-details/675c85b2085116a13353ce8b 

Copyright (c) 2025 Ventus Therapeutics U.S., Inc.


# 2. Getting started

Minimum of Python 3.6 is required.

## Installation
1. Download or git clone using: https://github.com/Ventus-Therapeutics/rapa.git
2. Run:  ``` pip install -r requirement.txt ``` or ``` conda install --yes --file requirement.txt ``` to install the dependencies.

The requirement.txt file can be found in the repository: https://github.com/Ventus-Therapeutics/rapa/blob/v1.0/docs/requirement.txt 


## Usage
### Preparing Files for RAPA (not supported by this repository)
#### RAPA requires that the input PDB file meets the following criteria
1. **No gaps** – Protein structure must be contiguous. There must be no missing residues or missing atoms.  
2. **No steric clashes** – The structure must not contain steric clashes.  
3. **No hydrogen atoms** – The PDB file must not contain hydrogen atoms.  
4. **AMBER format** – All residue names must follow the AMBER format.  

---
### Running RAPA  

#### Basic Usage  
RAPA is executed with a PDB file as input. For example, to run RAPA on the PDB file `1xl2.pdb`:  

```  
rapa.py -pID [input_pdb_name] -o [out_name]   
rapa.py -pID '1xl2' -o '1xl2_out'  
```

#### Command-Line Arguments  

##### `-pID` or `--protein_id` (Required)  
- **Type:** String  
- **Description:** Specifies the input PDB file name. RAPA **will not run** without this argument.  
- **Example:**  
  ```  
  rapa.py -pID '1bcd'  
  ```

##### `-o` or `--out_name` (Optional)  
- **Type:** String  
- **Default:** `[pID]_out`  
- **Description:** Defines the prefix for the output PDB files.  
- **Example:**  
  ```bash  
  rapa.py -pID '1bcd' -o '1bcd_o'  
  ```  
  This will generate output files: `1bcd_o_0.pdb`, `1bcd_o_1.pdb`, `1bcd_o_2.pdb`, etc.  
  If omitted, RAPA defaults to `1bcd_out_0.pdb`, `1bcd_out_1.pdb`, etc.  

Since `-o` is optional, the following command will also run RAPA:  
```bash  
rapa.py -pID '1xl2'  
```

##### Listing Available Options  
Use the `-h` flag to view all available command-line options:  
```bash  
rapa.py -h  
```

---
# 3. Files and Folders Generated by RAPA  

## 1. `[input_pdb_name]_HLPsp2.pdb`  
- Created for each input PDB.  
- Contains sp2 hydrogen and lone pair coordinates computed and added.  

## 2. `outputs_[input_pdb_name]` (Output Folder)  
This folder contains all output files generated by RAPA.  

### 2.1 `[input_pdb_name].info` (Run Information File)  
This file contains:  
- **Basic information:**  
  - PDB file name, ID, number of models, and chains.  
- **Residue and atom details:**  
  - Lists ASP/GLU residues investigated by RAPA.  
  - Displays unresolved ASP/GLU residues with their residue numbers, atom IDs, and distance details.  
      - If more than 2 ASP/GLU residues are within hydrogen bonding distance, RAPA does **not** resolve them. Manual intervention is required.  
- **Run summary:**  
  - Execution time.  
  - Number of output PDB files produced.  
  - File path of generated outputs.  

### 2.2 `pdb_out_[input_pdb_name]` (PDB Output Folder)  
Contains all PDB output files generated by RAPA.

---
# 4. Example Runs  
Test PDB files are available in the `tests` folder for validations.

---
# 5. Copyright and license
Code released under the [MIT License](https://github.com/Ventus-Therapeutics/rapa/blob/main/LICENSE)


MIT License
 

Copyright (c) 2025 Ventus Therapeutics U.S., Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

 

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


# 7. Authors:
This code is developed and released by Ventus Therapeutics U.S., Inc.
In conjunction with Tom Kurtzman's group at CUNY.
