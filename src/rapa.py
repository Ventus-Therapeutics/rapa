#!/usr/bin/env python
"""
Main file to run rapa

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


"""

import argparse
import sys, os
import timeit
import shutil
from Bio.PDB import *

import my_constants as mc
import setup_protein as stp
import hydrogen_placement_sp2 as hsp2
import state_assignment as sa


if __name__ == "__main__":

#############################input arguments#######################################################################
    #Creating an object parser which would help add the optional command line arguments 
    parser = argparse.ArgumentParser(description = 'basic argument')
    parser.add_argument( '-pID', '--protein_id', type= str, required = True, metavar= '', help='Input protein ID name. It is a required argument of data type string.')

    parser.add_argument( '-o', '--out_name', type = str, metavar= '', help='Prefix of output file name. It is an optional argument of data type string. If not provided, it will use the default value as "[pID]_out". ')
    
    parser.add_argument( '-s', '--single_pdb_out', type = int, metavar= '', help='This flag requests the program to provide a single output pdb file. This is done by removing the degeneracy of unknown residues.')

    parser.add_argument( '-l', '--log_file', type = int, metavar= '', help='Log flag creates a .log_file file which contains details of the run. It is an optional argument of data type integer with default value 0.')

    parser.add_argument( '-d', '--debug', type = int, metavar= '', help='Debug flag creates a .debug file which contains additional details of the run. It is an optional argument of data type integer with default value 0.')


    parser.add_argument( '-k_hlp', '--keep_hlp', type = int, metavar= '', help=' A integer flag with value 0 or 1 to provide an option if the additional PDB created with the added hydrogen and lone pair coordinates (with suffix "_HLPsp2.pdb") needs to be generated as an output. The default value for this flag is 0.')

    parser.add_argument( '-hlp', '--HLPsp2_known', type = int, metavar= '', help='A flag with data type integer value of 0 or 1 to indicate if the additional PDB (having suffix "_HLPsp2.pdb"), with SP2 hydrogen and lone pair coordinates exist. This can occur if the user requested the program to generate this additional PDB by providing "-k_hlp 1" as a flag in the previous run for the same PDB. Note, it is assumed RAPA for the given PDB is needed to run twice. During the first run the user provided the "-k_hlp" flag and generated the additional PDB (with suffix "_HLPsp2.pdb"). This was then used in the second run by providing "-hlp 1" in the second run. This two step approach is recommended for debugging as it stores an intermediate calculation and saves computational time. It is an optional argument with default value 0. ')

    parser.add_argument( '-v', '--version', action='version', version='%(prog)s 1.0 ')

    args = parser.parse_args()
    a=1
    #Protein id
    if(args.protein_id):
        protID = args.protein_id
    else:
        protID = ''
        sys.exit('please provide an input PDB file name')
    
    #if hydrogen bonded to sp2 side chain atoms of unknown residue, along with lone pair locations are known or not.
    #If the tool is run once-it will create a PDB file with hydrogen bonded with sp2 atoms, and lone pair coordinates. The code will use that if you give a value of 1 for this flag
    if(args.keep_hlp):
        keep_hlp = args.keep_hlp
    else:
        keep_hlp = 0
    
    if(args.HLPsp2_known):
        HLPsp2_known = args.HLPsp2_known
    else:
        HLPsp2_known = 0

    #the out files will be created with XXX_Number.pdb where XXX will be the name provided by the user, otherwise default is: protID_'out'
    if(args.out_name):
        fOutName = args.out_name
    else:
        fOutName = protID + '_out' 
 
    #Switch on debug mode
    if(args.debug):
        debug = args.debug
    else:
        debug = 0
 
    if(args.log_file):
        log_file = args.log_file
    else:
        log_file = 0

    if(args.single_pdb_out):
        mc.ECutOff=0
    else:
        mc.ECutOff=1

    #to have protID accessible from any point in the code 
    mc.protID = protID

    if(debug):
        #debug file
        stp.start_debug_file(__name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debug_file_name()
        fd = open(fDebugName, "w")
   
    #basic information file
    fInfoName =  stp.get_info_file_name()
    fInfo = open(fInfoName, "w")

    if(log_file):
        #log_file file
        fLogName = stp.get_log_file_name()
        f = open(fLogName, "w")

   
    starttime = timeit.default_timer()

    if(log_file):
        f.write(f"The PDB ID being used: {protID} and start time is: {starttime}\n")
        f.flush()

    fInfo.write(f"PDB ID: {protID} \n")
    fInfo.flush()
    
    if(debug):
        fd.write(f"PDB ID: {protID} \n")
        fd.flush()

    #set up structure, know the number of model and chain in the pdb 
    structure = stp.setup_structure(protID, outFolder = '.', fName = None)
    chains = [item for sublist in structure for item in sublist]
    numModels = len(structure.child_list)
    numChains = len(chains)


    if(log_file):
        f.write(f"\nThe structure has number of models {numModels}: {structure.child_list} and number of chains: {numChains}: {chains}\n\n")
        f.flush()
        f.close()

    fInfo.write(f"Models: {numModels}: {structure.child_list}\n")
    fInfo.write(f"Chains: {numChains}: {chains}\n\n")
    fInfo.flush()
    fInfo.close()

    if(debug):
        fd.write(f"Models: {numModels}: {structure.child_list}\n")
        fd.write(f"Chains: {numChains}: {chains}\n\n")
        fd.flush()
        fd.close()
        
          
    #rename HID/HIE/HIP/ASH/GLH->HIS/HIS/HIS/ASP/GLU
    stp.change_HIDEP_ASH_GLH(structure, debug=debug)

    
    #initialize the known/unknown residues(ASN/GLN/HIS, and mark if 2 ASPs/GLUs are in h-bonding distance to each other)
    stp.set_initial_known_residues_and_rotamers(structure, debug=debug)

    mc.xc_orig, mc.yc_orig, mc.zc_orig = stp.get_centroid(structure,debug=debug)
    #normalize so the new centroid=(0,0,0)
    stp.normalize_atom_coords(structure, debug=debug)

###########################Adding hydrogen and  connected to sp2 side chain atoms of unknown residue ##########################
    if(HLPsp2_known == 0):
        lastSerial = list(structure.get_residues())[-1].child_list[-1].serial_number 
        #This does not use neighboring residues
        lastSerial,_  = hsp2.place_lonepair_on_backbone(structure, lastSerial, debug=debug)
        lastSerial,_ = hsp2.add_sp2_sidechain_lonepairs(structure,lastSerial, debug=debug)
        lastSerial,_ = hsp2.placeHydrogens_backbone(structure, lastSerial, log_file=log_file, debug=debug)
        #Adding hydrogen to side chains:
        lastSerial,_ = hsp2.add_sp2_sidechain_hydrogens(structure,lastSerial, log_file=log_file, debug=debug)
        fName = f'{protID}_HLPsp2.pdb'
        stp.write_to_PDB(structure, fName, removeHLP = False, removeHall = False, set_original_centroid=False, log_file=log_file, debug=debug) 


        if(log_file):
            f = open(fLogName, "a")
            f.write("After adding the hydrogens..\n")
            f.flush()
            f.close()

        if(debug):
            fd = open(fDebugName, "a")
            fd.write("After adding the hydrogens..\n")
            fd.flush()
            fd.close()

        protIDName = protID + "_HLPsp2"

    else: 
        protIDName = protID + "_HLPsp2"

        if(log_file):
            f = open(fLogName, "a")
            f.write("\nHydrogen and lone pairs were already present!..\n")
            f.flush()
            f.close()

        if(debug):
            fd = open(fDebugName, "a")
            fd.write("\nHydrogen and lone pairs were already present!..\n")
            fd.flush()
            fd.close()

     
    if(debug):
        outputFolder= stp.get_output_folder_name()
        opFolder = f"./{outputFolder}/energyInfo_{mc.protID}/"
        checkFolderPresent = os.path.isdir(opFolder)
        if checkFolderPresent: 
            shutil.rmtree(opFolder)
    
    #setting up the new structure with hydrogen and lone pairs
    structure = stp.setup_structure(protIDName,outFolder = '.', fName = None)
    stp.set_initial_known_residues_and_rotamers(structure, debug=debug)

    #flagging the unknown ASP/GLUs
    unknownASP,unknownASP_atomInfo = stp.get_all_unknown_ASP_GLU(structure, resName = 'ASP', log_file=log_file, debug=debug)
    unknownGLU,unknownGLU_atomInfo = stp.get_all_unknown_ASP_GLU(structure, resName = 'GLU',log_file=log_file, debug=debug)
    over2ASPs = 0
    over2GLUs = 0
    if(unknownASP):
        structure,uniqueIDsASP,over2ASPs = stp.accomodate_for_ASPs_GLUs( structure, unknownASP, unknownASP_atomInfo, log_file=log_file, debug=debug)
    if(unknownGLU):
        structure, uniqueIDsGLU, over2GLUs = stp.accomodate_for_ASPs_GLUs( structure, unknownGLU, unknownGLU_atomInfo, log_file=log_file, debug=debug)

    #Flag the side chain residue hydrogen position for SER, THR, LYS, TYR is unknown.
    stp.set_initial_residue_side_chain_hydrogen_unknown(structure, debug=debug)

    maxLevel=15 #maximum vertical depth level it can go down while branching
    ##For a given structure-how many times we go over unknown residue list till we reach no change from unknown to known(in the list of unknown residues) for the given structure. 
    chValMax=15

    #resolve the ambiguities in a given structure
    fileNums,filesGen, LOS,skipInfoAll, skipValAll = sa.resolve_residue_ambiguities_in_structure(structure, fOutName, maxLevel, chValMax, set_original_centroid=True, log_file=log_file, debug=debug)

    netTime = timeit.default_timer() - starttime

    if(log_file):
        f = open(fLogName, "a")

    fInfo = open(fInfoName, "a")


    if(log_file):
        f.write(f"###############################################################\n")
        f.write(f"###############################################################\n")
        f.write(f"###############################################################\n\n\n")
        f.write(f"The time taken is :{netTime} seconds or {netTime/60} minutes or {netTime/3600} hours \n")
        f.write(f"Number of files generated: {fileNums} and files generated:\n{filesGen}\n")
        f.flush()


    
    fInfo.write(f"###############################################################\n")
    fInfo.write(f"###############################################################\n")
    fInfo.write(f"###############################################################\n\n\n")
    fInfo.write(f"The time taken is :{netTime} second or {netTime/60} mins or  {netTime/3600} hours\n")
    fInfo.write(f"Number of files generated: {fileNums} and files generated:\n{filesGen}\n")
    fInfo.flush()

    if(debug):
        fd = open(fDebugName, "a")
        fd.write(f"###############################################################\n")
        fd.write(f"###############################################################\n")
        fd.write(f"###############################################################\n\n\n")
        fd.write(f"The time taken is :{netTime} seconds or {netTime/60} minutes or {netTime/3600} hours \n")
        fd.write(f"Number of files generated: {fileNums} and files generated:\n{filesGen}\n")
        fd.flush()

    if(over2ASPs>0 or over2GLUs>0):  
        if(log_file):
            f.write(f"WARNING: There are more than 2 ASPs/GLUs.\n\n over2ASPs: {over2ASPs}\n over2GLUs: {over2GLUs}. Human intervention is required. Look for WARNING in log_file file for details\n")
            f.flush()

        fInfo.write(f"WARNING: There are more than 2 ASPs/GLUs.\n\n over2ASPs: {over2ASPs}\n over2GLUs: {over2GLUs}. Human intervention is required. Look for WARNING in log_file file for details\n")
        fInfo.flush()
        
        if(debug):
            fd = open(fDebugName, "a")
            fd.write(f"WARNING: There are more than 2 ASPs/GLUs.\n\n over2ASPs: {over2ASPs}\n over2GLUs: {over2GLUs}. Human intervention is required. Look for WARNING in log_file file for details\n")
            fd.flush()

    if(not keep_hlp):
        file_hlp=f"{protID}" + "_HLPsp2.pdb"
        if os.path.exists(file_hlp):
            os.remove(file_hlp) 
        else:
            if(debug):
                fd = open(fDebugName, "a")
                fd.write(f"File {file_hlp} does not exist.")
                fd.push()

    if(log_file):
        f.close()
   
    if(debug):
        stp.end_debug_file(__name__,sys._getframe().f_code.co_name, fd)

    print("************************************************************") 
    print(f"RAPA exiting. Run for the given PDB: {protID} is completed")
    print("************************************************************") 
    sys.exit()






