#!/usr/bin/env python

import argparse
import sys,os
import Bio, scipy
import numpy as np
import code

from Bio.PDB import *
import timeit


import myMath as mm
import myResAtom as mra
import myConstants as mc
import closeAtoms as cats
import setupProt as stp
import hPlacementSP2 as hsp2
import hPlacementSP3_energy as hsp3
import stateAssignment as sa
import shutil


if __name__== "__main__":

#############################input arguments#######################################################################
    #Creating an object parser which would help add the optional command line arguments 
    parser = argparse.ArgumentParser(description = 'basic argument')
    parser.add_argument('-pID','--proteinID', type = str, metavar = '', help='Protein ID name')

    parser.add_argument('-o','--fileOutName', type = str, metavar = '', help='output file name')
    parser.add_argument('-hlp','--HLPsp2_known', type = int, metavar = '', help='sp2 hydrogens and lone pairs in a pdb is already present')
    parser.add_argument('-d','--debug', type = int, metavar = '', help='debug flag to view more details')


    args = parser.parse_args()
    
    #Protein id
    if(args.proteinID):
        protID = args.proteinID
    else:
        protID = '6loc'
    
    #if hydrogens bonded to sp2 side chain atoms of unknown residue, along with lone pair locations are known or not. If the tool is run once-it will create a pdb file with hydrogens bonded with sp2 atoms, and lone pair coordinates. The code will use that if you give a value of 1 for this flag
    if(args.HLPsp2_known):
        HLPsp2_known = args.HLPsp2_known
    else:
        HLPsp2_known = 0

    #the out files will be create with proteinid_XXX_Number.pdb where XXX will be the name provided here
    if(args.fileOutName):
        fOutName = args.fileOutName
    else:
        fOutName = protID + '_out' 
 
    #if you want to enter the debug mode
    if(args.debug):
        debug = args.debug
    else:
        debug = 0
########################################################################################################

    #to have protID accessible from any point in the code 
    mc.protID = protID

    if(debug ==1):
        #debug file
        stp.startDebugFile(__name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debugFileName(debug=debug)
        fd = open(fDebugName, "w")
   
    #basic information file
    fInfoName =  stp.get_infoFileName(debug=debug)
    fInfo = open(fInfoName, "w")

    #log file
    fLogName = stp.get_logFileName(debug=debug)
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
    structure = stp.setupStructure(protID, outFolder = '.', fName = None, debug=debug) 
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
    stp.set_initialKnownAndRotamers(structure,debug=debug)
    #normalize so the new centroid=(0,0,0)
    stp.normalizeAtomCoords(structure, debug=debug) 

    xc_new, yc_new, zc_new = stp.get_centroid(structure, debug=debug)

###########################Adding hydrogens connected to sp2 side chain atoms of unknown residue ##########################
    if(HLPsp2_known == 0):
        lastSerial = list(structure.get_residues())[-1].child_list[-1].serial_number 
        #This does not use neighboring residues
        lastSerial, bbLPcoords = hsp2.placeLP_BB(structure, lastSerial, debug=debug)
        lastSerial, bbLPcoords = hsp2.addSP2SideChainLPs(structure,lastSerial, debug=debug)
        lastSerial, bbHcoords = hsp2.placeHydrogens_BB(structure, lastSerial, debug=debug)   
        ###Adding hydrogens to side chains!!:
        lastSerial, hCoords = hsp2.addSP2SideChainHydrogens(structure,lastSerial, debug=debug)
        fName = f'{protID}_HLPsp2.pdb'
        stp.write2PDB(structure, fName, removeHLP = False, debug=debug) 

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
        outputFolder= stp.get_outputFolderName(debug=debug)
        opFolder = f"./{outputFolder}/energyInfo_{mc.protID}/"
        checkFolderPresent = os.path.isdir(opFolder)
        if checkFolderPresent: 
            shutil.rmtree(opFolder)

    #setting up the new structure with hydrogens and lone pairs
    structure = stp.setupStructure(protIDName,outFolder = '.', fName = None, debug=debug) 
    stp.set_initialKnownAndRotamers(structure, debug=debug)

    #flagging the unknown ASP/GLUs
    unknownASP,unknownASP_atomInfo = stp.get_allUnknownASP_GLU(structure, resName = 'ASP', debug=debug)
    unknownGLU,unknownGLU_atomInfo = stp.get_allUnknownASP_GLU(structure, resName = 'GLU', debug=debug)
    over2ASPs = 0
    over2GLUs = 0
    if(unknownASP):
        structure,uniqueIDsASP,over2ASPs = stp.accomodateForASP_GLUs( structure, unknownASP, unknownASP_atomInfo, debug=debug)
    if(unknownGLU):
        structure, uniqueIDsGLU, over2GLUs = stp.accomodateForASP_GLUs( structure, unknownGLU, unknownGLU_atomInfo, debug=debug)

    #Identify which side chain residue hydrogen (SER, THR, LYS, TYR) is known or not
    stp.set_initialResSC_hydKnown(structure, debug=debug)

    
    maxLevel=15 #maximum vertical depth level it can go down
    ##For a given structure-how many times we go over unknown residue list till we reach no change from unknown to known(in the list of unknown residue) for the given structure. 
    chValMax=15 
    #resolve the ambiguities in a given structure
    fileNums,filesGen, LOS,skipInfoAll, skipValAll = sa.resolveResidueAmbiguityInStructure(structure, fOutName, maxLevel, chValMax, debug)
    netTime = timeit.default_timer() - starttime

    
    #As the program ends-not the time took, number of files generated and if the user needs to take care of any ASPs/GLUs
    f = open(fLogName, "a")
    fInfo = open(fInfoName, "a")

    f.write(f"###############################################################\n")
    f.write(f"###############################################################\n")
    f.write(f"###############################################################\n\n\n")
    f.write(f"The time taken is :{netTime} seconds or {netTime/60} minutes or {netTime/3600} hours \n")
    f.write(f"Num of files generated: {fileNums} and files generated:\n{filesGen}\n")
    f.flush()
 
    fInfo.write(f"The time taken is :{netTime} second or {netTime/60} mins or  {netTime/3600} hours\n")
    fInfo.write(f"Num of files generated: {fileNums} and files generated:\n{filesGen}\n")
    fInfo.flush()

    if(debug==1):
        fd = open(fDebugName, "a")
        fd.write(f"###############################################################\n")
        fd.write(f"###############################################################\n")
        fd.write(f"###############################################################\n\n\n")
        fd.write(f"The time taken is :{netTime} seconds or {netTime/60} minutes or {netTime/3600} hours \n")
        fd.write(f"Num of files generated: {fileNums} and files generated:\n{filesGen}\n")
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
        stp.endDebugFile(__name__,sys._getframe().f_code.co_name, fd)

    sys.exit()







