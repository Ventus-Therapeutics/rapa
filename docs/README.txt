PARA or the protonation and rotameric assignment tool helps in protein preparation to assign energetically favorable protonation and rotameric states. 

Please make sure of the following:
1.PDB does not have any gaps.
2.


To run this tool, the main file is PRAT.py. 
The input flags to be added:
1) -pID: is a string of protein ID

2) -HLPsp2_known: is an integer value 0 or 1 to indicate if the Hydrogen and Lone pair coordinates are already added in the PDB file. If you are running this tool for the first time, provdie the value 0, else there would be a file with suffix "_HLPsp2" in your folder. If the file with the suffix "_HLPsp2" is already present in your folder then provide the integer value: 1. This will help save some compute time.

3) -o: provides part of the suffix for the output pdb file. For example if we provide: -o '1bcd_out' then the output pdb files will be: 1bcd_out_0, 1bcd_out_1, 1bcd_out_2, 1bcd_out_3,...etc

For example to a run the tool from the current folder where the PDB file is present, and the PRAT_libs folder is one directory level above do the following:

python3 ../RAPA_libs/SRC/rapa.py -pID '1xl2_spruce_noH' -HLPsp2_known 0 -o '1xl2_out'
or (if you have information about Hydrogen and lone pairs):
python3 ../RAPA_libs/SRC/rapa.py -pID '1xl2_spruce_noH' -HLPsp2_known 1 -o '1xl2_out'
