#!/usr/bin/env python

import argparse
import sys,Bio, scipy
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



if __name__== "__main__":
    
    #Creating an object parser which would help add the optional command line arguments 
    parser = argparse.ArgumentParser(description = 'basic argument')
    parser.add_argument('-pID','--proteinID', type = str, metavar = '', help='Protein ID name')

    parser.add_argument('-o','--fileOutName', type = str, metavar = '', help='output file name')
    parser.add_argument('-HLPsp2_known','--HLPsp2_known', type = int, metavar = '', help='sp2 hydrogens and lone pairs in a pdb is already present')
    parser.add_argument('-d','--debug', type = int, metavar = '', help='debug flag to view more details')


    args = parser.parse_args()
   
    if(args.proteinID):
        protID = args.proteinID
    else:
        protID = '6loc'

    if(args.HLPsp2_known):
        HLPsp2_known = args.HLPsp2_known
    else:
        HLPsp2_known = 0

    if(args.fileOutName):
        fOutName = args.fileOutName
    else:
        fOutName = protID + '_out' 
  
    if(args.debug):
        debug = args.debug
    else:
        debug = 0

   
    mc.protID = protID
   
    fInfoName =  stp.get_infoFileName(protID)
    fInfo = open(fInfoName, "w")

    fLogName = stp.get_logFileName(protID)
    f = open(fLogName, "w")

    if(debug ==1):
        fDebugName = stp.get_debugFileName(protID)
        fd = open(fDebugName, "w")

    starttime = timeit.default_timer()
    f.write(f"The PDB ID being used: {protID} and start time is: {starttime}\n")
    f.flush()

    fInfo.write(f"PDB ID: {protID} \n")
    fInfo.flush()
    
    if(debug ==1):
        fd.write(f"PDB ID: {protID} \n")
        fd.flush()
        fd.close()

    
    
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

    stp.set_initialKnownAndRotamers(structure)
    stp.normalizeAtomCoords(structure) 

    xc_new, yc_new, zc_new = stp.get_centroid(structure)


    mc.allActiveAtoms, mc.allDonors, mc.allAcceptors, mc.allBoth, mc.allTBD, mc.allOnlyDonors, mc.allOnlyAcceptors, mc.allOnlyDonorsAcceptors, mc.allDonorsAcceptorsBoth = stp.get_allActiveAtomsInfo(structure)

############################Adding hydrogens...############################################################
###########################################################################################################################
    if(HLPsp2_known == 0):
        lastSerial = list(structure.get_residues())[-1].child_list[-1].serial_number
        
        #This does not use neighboring residues
        lastSerial, bbLPcoords = hsp2.placeLP_BB(structure, lastSerial, debug = 0)
        lastSerial, bbLPcoords = hsp2.addSP2SideChainLPs(structure,lastSerial, debug =0)
        lastSerial, bbHcoords = hsp2.placeHydrogens_BB(structure, lastSerial, debug = 0)   
    ###ADDING Hydrogens to side chains!!:
        lastSerial, hCoords = hsp2.addSP2SideChainHydrogens(structure,lastSerial, debug =0)


        fName = f'{protID}_HLPsp2.pdb'
        stp.write2PDB(structure, fName, removeHLP = False) 

        f = open(fLogName, "a")
        f.write("After adding the hydrogens..\n")
        f.close()
        protIDName = protID + "_HLPsp2"

    else: 
        protIDName = protID + "_HLPsp2"
        f = open(fLogName, "a")
        f.write("\nHydrogen and lone pairs were already present!..\n")
        f.close()
      
    structure = stp.setupStructure(protIDName,outFolder = '.', fName = None, debug=debug) 
    stp.set_initialKnownAndRotamers(structure, debug)

    #code.interact(local = locals())
    unknownASP,unknownASP_atomInfo = stp.get_allUnknownASP_GLU(structure, resName = 'ASP', debug=debug)
    unknownGLU,unknownGLU_atomInfo = stp.get_allUnknownASP_GLU(structure, resName = 'GLU', debug=debug)
  
    #################################################################################################################
    over2ASPs = 0
    over2GLUs = 0
    if(unknownASP):
        structure,uniqueIDsASP,over2ASPs = stp.accomodateForASP_GLUs( structure, unknownASP, unknownASP_atomInfo)
    if(unknownGLU):
        structure, uniqueIDsGLU, over2GLUs = stp.accomodateForASP_GLUs( structure, unknownGLU, unknownGLU_atomInfo)

    #Identify which side chain residue is known or not
    stp.set_initialResSC_hydKnown(structure)


    maxLevel =15
    chValMax = 15

    fileNums,filesGen, LOS,skipInfoAll, skipValAll = sa.resolveResidueAmbiguityInStructure(structure, fOutName, maxLevel, chValMax, debug)
    netTime = timeit.default_timer() - starttime


    f = open(fLogName, "a")
    fInfo = open(fInfoName, "a")

    f.write(f"###############################################################\n")
    f.write(f"###############################################################\n")
    f.write(f"###############################################################\n\n\n")
    f.write(f"The time taken is :{netTime} seconds or {netTime/60} minutes \n")
    f.write(f"Num of files generated: {fileNums} and files generated:\n{filesGen}\n")
    f.flush()
 
    fInfo.write(f"The time taken is :{netTime} second or {netTime/60} mins \n")
    fInfo.write(f"Num of files generated: {fileNums} and files generated:\n{filesGen}\n")
    fInfo.flush()

    if(over2ASPs>0 or over2GLUs>0):  
        f.write(f"WARNING: There are more than 2 ASPs/GLUs.\n\n over2ASPs: {over2ASPs}\n over2GLUs: {over2GLUs}. Human intervention is required. Look for WARNING in log file for details\n")
        fInfo.write(f"WARNING: There are more than 2 ASPs/GLUs.\n\n over2ASPs: {over2ASPs}\n over2GLUs: {over2GLUs}. Human intervention is required. Look for WARNING in log file for details\n")
        f.flush()
        fInfo.flush()

    f.flush()
    f.close()

    sys.exit()







