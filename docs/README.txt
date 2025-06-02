RAPA or the Rotamer And Protonation Assignment tool helps in protein preparation to assign energetically favorable protonation and rotameric states for a given protein structure.

Prerequisite for the input pdb file is that the structure in the pdb must not have any gaps. To run this tool, the main file is rapa.py. 

The input flags to be added are the following:
    1) -pID (an optional argument-but please provide this, default value='6loc'): is a string of protein ID or the pdb input file you would like to provide. It is set as an optional argument-so if you do not provide this flag, the program will look for the following pdb file: 6loc.pdb

    2) -hlp (an optional argument, default value=0): is an integer value 0 or 1 to indicate if hydrogen and lone pair coordinates are already added in the pdb file. If you are running this tool for the first time, provide the value 0(which is also a default value). If you already ran the code once for a particular protein then there would be a file with suffix "_HLPsp2" in your folder. If the file with the suffix "_HLPsp2" is already present in your folder then you can provide the integer value: 1. This will help save some compute time.

    3) -o (an optional argument, default value='pID_out'): provides part of the suffix for the output pdb file. For example if we provide: -o '1bcd_out' then the output pdb files will be: 1bcd_out_0, 1bcd_out_1, 1bcd_out_2, 1bcd_out_3,...etc

For example to a run the tool from the current folder where the PDB file is present, and the PRAT_libs folder is one directory level above do the following.Assuming we have a pdb: '1xl2_spruce_noH.pdb':

python3 ../RAPA_libs/src/rapa.py -pID '1xl2_spruce_noH' -hlp 0 -o '1xl2_out'

or (if you have information about Hydrogen and lone pairs):
python3 ../RAPA_libs/src/rapa.py -pID '1xl2_spruce_noH' -hlp 1 -o '1xl2_out'

4) -d: is the debug flag. If debug flag is switched on then it will produce a .debug file. This file will have additional detailed information. Please note-on switching on this, the code will become extremely slow. Please use it only for debug purposes.

example:
$../rapa/src/rapa.py -pID 1bcd_spruce_noH -o 1bcd_out -hlp 0 -d 1
