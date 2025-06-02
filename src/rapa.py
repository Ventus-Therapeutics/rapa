#!/usr/bin/env python


###########################################################################################################################################################################
##This is the main file to run rapa and the following steps are taken to setup and run
##1.Set up protein structure/parser.
##2.If HIE/HID/HIP/ASH/GLH is present in pdb and fix to HIS/HIS/HIS/ASP/GLU.
##3.Flag the known residues. 
##4.Normalize the coordinates.
##5.Add sp2 hydrogens and lone pairs.
##6.Flag unknown ASPs and GLUs.
##7.Flag unknown side chain hydrogens of SER/THR/LYS/TYR.
##8.Resolve the ambiguities in structure.
############################################################################################################################################################################

import argparse
import sys,os
import Bio, scipy
import numpy as np
import code

from Bio.PDB import *
import timeit


import my_math as mm
import my_residue_atom as mra
import my_constants as mc
import close_atoms as cats
import setup_protein as stp
import hydrogen_placement_sp2 as hsp2
import hydrogen_placement_sp3 as hsp3
import state_assignment as sa
import shutil


if __name__== "__main__":

#############################input arguments#######################################################################
    #Creating an object parser which would help add the optional command line arguments 
    parser = argparse.ArgumentParser(description = 'basic argument')
    parser.add_argument('-pID','--protein_id', type = str,required = True, metavar = '', help='Input protein ID name. It is a required argument of data type string.')

    parser.add_argument('-o','--out_name', type = str, metavar = '', help='Prefix of output file name. It is an optional argument of data type string. If not provided, it will use the default value as "[pID]_out". ')
    parser.add_argument('-hlp','--HLPsp2_known', type = int, metavar = '', help='A flag with data type integer value of 0 or 1 to indicate if sp2 hydrogen and lone pair coordinates have been added in a new pdb file with suffix "_HLPsp2.pdb" for the given input structure. If the user is running this tool for the first time for a particular protein, the user must provide the value 0(which is also the default value) as the new pdb file (with suffix "_HLPsp2.pdb") will not be present. However if the file with suffix "_HLPsp2.pdb" for the input pdb is already present, then the value 1 for this flag can be given as it will help save some computation time. It is an optional argument of data type integer with default value 0.')
    parser.add_argument('-d','--debug', type = int, metavar = '', help='Debug flag creates a .debug file which contains additional details of the run. It is an optional argument of data type integer with default value 0.')

    parser.add_argument('-v','--version', action='version', version='%(prog)s 0.6r1 ')

    args = parser.parse_args()
    
    #Protein id
    if(args.protein_id):
        protID = args.protein_id
    else:
        protID = ''
        sys.exit('please provide an input pdb file name')
    
    #if hydrogens bonded to sp2 side chain atoms of unknown residue, along with lone pair locations are known or not. If the tool is run once-it will create a pdb file with hydrogens bonded with sp2 atoms, and lone pair coordinates. The code will use that if you give a value of 1 for this flag
    if(args.HLPsp2_known):
        HLPsp2_known = args.HLPsp2_known
    else:
        HLPsp2_known = 0

    #the out files will be create with XXX_Number.pdb where XXX will be the name provided by the user, otherwise default is: protID_'out'
    if(args.out_name):
        fOutName = args.out_name
    else:
        fOutName = protID + '_out' 
 
    #if you want to enter the debug mode
    if(args.debug):
        debug = args.debug
    else:
        debug = 0

    ###########################################################################################################################################################################
    ###########################################################################################################################################################################

    #to have protID accessible from any point in the code 
    mc.protID = protID

    if(debug ==1):
        #debug file
        stp.start_debug_file(__name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debug_file_name(debug=debug)
        fd = open(fDebugName, "w")
   
    #basic information file
    fInfoName =  stp.get_info_file_name(debug=debug)
    fInfo = open(fInfoName, "w")

    #log file
    fLogName = stp.get_log_file_name(debug=debug)
    f = open(fLogName, "w")

   
    starttime = timeit.default_timer()
    f.write(f"The PDB ID being used: {protID} and start time is: {starttime}\n")
    f.flush()

    fInfo.write(f"PDB ID: {protID} \n")
    fInfo.flush()
    
    if(debug ==1):
        fd.write(f"PDB ID: {protID} \n")
        fd.flush()

    #set up structure, know the number of model and chain in the pdb 
    structure = stp.setup_structure(protID, outFolder = '.', fName = None, debug=debug) 
    chains = [item for sublist in structure for item in sublist]
    numModels = len(structure.child_list)
    numChains = len(chains)

    f.write(f"\nThe structure has number of models {numModels}: {structure.child_list} and number of chains: {numChains}: {chains}\n\n")
    f.flush()
    f.close()

    fInfo.write(f"Models: {numModels}: {structure.child_list}\n")
    fInfo.write(f"Chains: {numChains}: {chains}\n\n")
    fInfo.flush()
    fInfo.close()

    if(debug==1):
        fd.write(f"Models: {numModels}: {structure.child_list}\n")
        fd.write(f"Chains: {numChains}: {chains}\n\n")
        fd.flush()
        fd.close()
        
          

    #if there is HIE/HID/HIP or ASH/GLH in pdb-that has the right protonation and rotameric state. Then remove this line!!
    stp.change_HIDEP_ASH_GLH(structure, debug=debug)

    
    #initialize the known/unknown residues(ASN/GLN/HIS, and mark if 2 ASPs/GLUs are in hbonding distance to each other)
    stp.set_initial_known_residues_and_rotamers(structure,debug=debug)
    #normalize so the new centroid=(0,0,0)
    stp.normalize_atom_coords(structure, debug=debug) 

    xc_new, yc_new, zc_new = stp.get_centroid(structure, debug=debug)

###########################Adding hydrogens and  connected to sp2 side chain atoms of unknown residue ##########################
    if(HLPsp2_known == 0):
        lastSerial = list(structure.get_residues())[-1].child_list[-1].serial_number 
        #This does not use neighboring residues
        lastSerial, bbLPcoords = hsp2.place_lonepair_on_backbone(structure, lastSerial, debug=debug)
        lastSerial, bbLPcoords = hsp2.add_sp2_sidechain_lonepairs(structure,lastSerial, debug=debug)
        lastSerial, bbHcoords = hsp2.placeHydrogens_backbone(structure, lastSerial, debug=debug)   
        ###Adding hydrogens to side chains!!:
        lastSerial, hCoords = hsp2.add_sp2_sidechain_hydrogens(structure,lastSerial, debug=debug)
        fName = f'{protID}_HLPsp2.pdb'
        stp.write_to_PDB(structure, fName, removeHLP = False, removeHall = False, debug=debug) 

        f = open(fLogName, "a")
        f.write("After adding the hydrogens..\n")
        f.flush()
        f.close()

        if(debug==1):
            fd = open(fDebugName, "a")
            fd.write("After adding the hydrogens..\n")
            fd.flush()
            fd.close()



        protIDName = protID + "_HLPsp2"

    else: 
        protIDName = protID + "_HLPsp2"
        f = open(fLogName, "a")
        f.write("\nHydrogen and lone pairs were already present!..\n")
        f.flush()
        f.close()

        if(debug==1):
            fd = open(fDebugName, "a")
            fd.write("\nHydrogen and lone pairs were already present!..\n")
            fd.flush()
            fd.close()

     
    if(debug==1):
        #Remove the folder if it is already present-so new files get written-instead of appending to old file
        outputFolder= stp.get_output_folder_name(debug=debug)
        opFolder = f"./{outputFolder}/energyInfo_{mc.protID}/"
        checkFolderPresent = os.path.isdir(opFolder)
        if checkFolderPresent: 
            shutil.rmtree(opFolder)

    #setting up the new structure with hydrogens and lone pairs
    structure = stp.setup_structure(protIDName,outFolder = '.', fName = None, debug=debug) 
    stp.set_initial_known_residues_and_rotamers(structure, debug=debug)

    #flagging the unknown ASP/GLUs
    unknownASP,unknownASP_atomInfo = stp.get_all_unknown_ASP_GLU(structure, resName = 'ASP', debug=debug)
    unknownGLU,unknownGLU_atomInfo = stp.get_all_unknown_ASP_GLU(structure, resName = 'GLU', debug=debug)
    over2ASPs = 0
    over2GLUs = 0
    if(unknownASP):
        structure,uniqueIDsASP,over2ASPs = stp.accomodate_for_ASPs_GLUs( structure, unknownASP, unknownASP_atomInfo, debug=debug)
    if(unknownGLU):
        structure, uniqueIDsGLU, over2GLUs = stp.accomodate_for_ASPs_GLUs( structure, unknownGLU, unknownGLU_atomInfo, debug=debug)

    #Flag the side chain residue hydrogen position for SER, THR, LYS, TYR is unknown.
    stp.set_initial_residue_side_chain_hydrogen_unknown(structure, debug=debug)

    
    maxLevel=15 #maximum vertical depth level it can go down while branching
    ##For a given structure-how many times we go over unknown residue list till we reach no change from unknown to known(in the list of unknown residues) for the given structure. 
    chValMax=15 
    #resolve the ambiguities in a given structure
    fileNums,filesGen, LOS,skipInfoAll, skipValAll = sa.resolve_residue_ambiguities_in_structure(structure, fOutName, maxLevel, chValMax, debug)
    netTime = timeit.default_timer() - starttime

    ###########################################################################################################################################################################
    ###########################################################################################################################################################################
    #As the program ends-note the time took, number of files generated and if the user needs to take care of any ASPs/GLUs
    f = open(fLogName, "a")
    fInfo = open(fInfoName, "a")

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

    if(debug==1):
        fd = open(fDebugName, "a")
        fd.write(f"###############################################################\n")
        fd.write(f"###############################################################\n")
        fd.write(f"###############################################################\n\n\n")
        fd.write(f"The time taken is :{netTime} seconds or {netTime/60} minutes or {netTime/3600} hours \n")
        fd.write(f"Number of files generated: {fileNums} and files generated:\n{filesGen}\n")
        fd.flush()

    if(over2ASPs>0 or over2GLUs>0):  
        f.write(f"WARNING: There are more than 2 ASPs/GLUs.\n\n over2ASPs: {over2ASPs}\n over2GLUs: {over2GLUs}. Human intervention is required. Look for WARNING in log file for details\n")
        fInfo.write(f"WARNING: There are more than 2 ASPs/GLUs.\n\n over2ASPs: {over2ASPs}\n over2GLUs: {over2GLUs}. Human intervention is required. Look for WARNING in log file for details\n")
        f.flush()
        fInfo.flush()
        
        if(debug==1):
            fd = open(fDebugName, "a")
            fd.write(f"WARNING: There are more than 2 ASPs/GLUs.\n\n over2ASPs: {over2ASPs}\n over2GLUs: {over2GLUs}. Human intervention is required. Look for WARNING in log file for details\n")
            fd.flush()


    f.flush()
    f.close()
   
    if(debug ==1):
        stp.end_debug_file(__name__,sys._getframe().f_code.co_name, fd)

    sys.exit()







