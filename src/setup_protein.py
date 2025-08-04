"""
Base functions for set up and run.


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

import sys
import os
import Bio
import copy

import numpy as np
import my_residue_atom as mra
import my_constants as mc
import global_config as gc

from Bio.PDB import *

def get_output_folder_name():
    
    """
    objective: To return name of the output folder
    O/P:
    -outputFolder: name of the output folder
    """
    #outputFolder = "outputs_"+mc.protID

    outputFolder = mc.out_name+"_outputs"

    return outputFolder


def get_log_file_name():

    """
    objective: To get the name of the log file
    Output:
    -fLogName: name of the log file
    """

    outputFolder = get_output_folder_name()

    fDest = f'{outputFolder}/log/'
    checkFolderPresent = os.path.isdir(fDest)    
    if not checkFolderPresent: os.makedirs(fDest)

    #fLogName = f"{fDest}"+f"{mc.protID}.log"

    fLogName = f"{fDest}"+f"{mc.out_name}.log"
    
    return fLogName

def get_info_file_name():
    """
    objective: To get the name of the info file name
    Output:
    -fInfoName: name of the info file
    """
        

    outputFolder = get_output_folder_name()

    fDest = f'{outputFolder}/'
    checkFolderPresent = os.path.isdir(fDest)    
    if not checkFolderPresent: os.makedirs(fDest)

    #fInfoName = f"{fDest}"+f"{mc.protID}.info"

    fInfoName = f"{fDest}"+f"{mc.out_name}.info"
    
    return fInfoName

def get_debug_file_name():
    
    """
    objective: To get the name of the info file name
    Output:
    -fDebugName: name of the info file
    """   

    outputFolder = get_output_folder_name()

    #fDest = f'{outputFolder}/'

    fDest = f'{outputFolder}/debug/'
    checkFolderPresent = os.path.isdir(fDest)    
    if not checkFolderPresent: os.makedirs(fDest)

    #fDebugName = f"{fDest}"+f"{mc.protID}.debug"

    fDebugName = f"{fDest}"+f"{mc.out_name}.debug"
    
    return fDebugName


#def get_pdb_out_folder():
#    
#    """
#    objective: To get the name of the folder where the output PDBs are stored
#    Output:
#    -foPDB: name of the pdb folder
#    """
#        
#
#    outputFolder = get_output_folder_name()
#
#    foPDB = f'{outputFolder}/pdb_out_{mc.protID}'
#    checkFolderPresent = os.path.isdir(foPDB)    
#    if not checkFolderPresent: os.makedirs(foPDB)
#
#    
#    return foPDB

def get_all_ASPs(structure):

    
    """
    objective: Get all ASPs in a given structure
    input: -structure: structure you need the ASP for
    output: allASPs: all ASP residues in a list data structure
    """
    
    allASPs = []

    for res in structure.get_residues():
        if(res.resname == 'ASP'):
            allASPs.append(res)

    return allASPs


def get_all_GLUs(structure):

    
    """
    objective: Get all GLU in a given structure
    input: -structure: structure you need the GLU for
    output: allGLUs: all GLU residues in a list data structure
    """    
    
    allGLUs = []
    for res in structure.get_residues():
        if(res.resname == 'GLU'):
            allGLUs.append(res)

    return allGLUs

def get_all_unknown_ASP_OD_atoms(structure):
    
    
    """
    objective: to get all OD1 and OD2 atoms of ASP 
    input:-structure: the protein structure to consider
    output:-OD1OD2_ASPatoms: list of atoms :OD1 and OD2 of all ASPs in a structure
    """
    

    allASPs = get_all_ASPs(structure)
    OD1OD2_ASPatoms = []

    ##Create allASP atom list for searching nbd!
    for res in allASPs:
        for at in res:
            if(at.id == 'OD1' or at.id =='OD2'):
                OD1OD2_ASPatoms.append(at)
    
    return OD1OD2_ASPatoms


def get_all_unknown_GLU_OE_atoms(structure):

    
    """
    objective: to get all OE1 and OE2 atoms of GLU
    input:-structure: the protein structure to consider
    output:-OE1OE2_GLUatoms: list of atoms :OE1 and OE2 of all GLUs in a structure
    """
    


    allGLUs = get_all_GLUs(structure)
    OE1OE2_GLUatoms = []

    ##Create allASP atom list for searching nbd!
    for res in allGLUs:
        for at in res:
            if(at.id == 'OE1' or at.id =='OE2'):
                OE1OE2_GLUatoms.append(at)
    
    return OE1OE2_GLUatoms


def get_all_unknown_ASP_GLU(structure, resName = 'ASP'):

    
    """
    objective: Get all the unknowns ASP or GLUs
    input:-structure: the structure in concern
          -resName:Name of the unknown residue that needs to be extracted

    output:-unknownASP_GLUatom: list of unknown atoms
           -unknownASP_GLUatomInfo: additional info including:-atom name, atom id, atom parent id, distance of Oxygen 1 of concern to oxygen in the h-bonding distance.
    """
    
    #create a list of atom to populate upon hitting an unknown residue with OD/OE atoms
    unknownASP_GLUatom = []
    unknownASP_GLUatomInfo = []

    if(gc.debug):
        print(f"searching for all Unknown res: {resName}")

    all_ASP_GLUs=[]
    ##check if any asp/glu is present. If not exit:
    for residue in structure.get_residues():
        if(residue.resname==resName):
            all_ASP_GLUs.append(residue)

    if(len(all_ASP_GLUs)==1 or len(all_ASP_GLUs)==0):
        if gc.debug:
            print(f"Looking for unknown ASPs/GLUs. However all ASPs/GLUs found are: {all_ASP_GLUs}")

        return unknownASP_GLUatom,unknownASP_GLUatomInfo
    #Get all the atoms that needs to be searched for an unknown ASP/GLU residue. Also get a list of all ASPs/GLUs to search
    if(resName == 'ASP'):
        if gc.log_file:
            print("Checking for unknown ASPs")

        searchASP_GLUatoms = get_all_unknown_ASP_OD_atoms(structure)
        allASPs_GLUs = get_all_ASPs(structure)
        oxygenName1 = 'OD1'
        oxygenName2 = 'OD2'

    else:
        if gc.log_file:
            print("Checking for unknown GLUs\n")

        searchASP_GLUatoms = get_all_unknown_GLU_OE_atoms(structure)
        allASPs_GLUs = get_all_GLUs(structure)
        oxygenName1 = 'OE1'
        oxygenName2 = 'OE2'

    if gc.debug:
        for atom in searchASP_GLUatoms:
            print(f"search atom: {atom} of {atom.parent} of chain: {atom.parent.parent}")
        for res in allASPs_GLUs:
            print(f"all ASP/GLU residues include: {res} of chain {res.parent}")
        
    for currASP_GLU in allASPs_GLUs:
        #remove side chain oxygen atoms of self residue so it is not counted within h-bond distance
        searchASP_GLUatoms.remove(currASP_GLU[oxygenName1])
        searchASP_GLUatoms.remove(currASP_GLU[oxygenName2])

        #incase the search list is empty. A special case when only one ASP/GLU is present in the entire PDB.
        if(searchASP_GLUatoms==[]):
            if gc.debug:
                print(f"Looking for unknown ASPs/GLUs. However search list is empty.")

            return unknownASP_GLUatom,unknownASP_GLUatomInfo

        #create the search list consisting of all unknown ASPs/GLUs
        ns = Bio.PDB.NeighborSearch(searchASP_GLUatoms)
    
        #use the search object/list  with respect to the two oxygen side chain atoms(OD1/OD2 or OE1/OE2) of the given ASP/GLU
        potUnknownAtom1 = ns.search(currASP_GLU[oxygenName1].coord,mc.deltaD)
        potUnknownAtom2 = ns.search(currASP_GLU[oxygenName2].coord,mc.deltaD)
        
        #if any of the unknown add it to the unknownASP_GLUatom list and collect relevant info
        if(potUnknownAtom1):
            for puat in potUnknownAtom1: 
                unknownASP_GLUatom.append(puat)
                unknownASP_GLUatomInfo.append([currASP_GLU[oxygenName1],currASP_GLU.id[1], puat, puat.parent.id[1], currASP_GLU[oxygenName1]-puat])
            if gc.debug:
                print(f"collecting unknownASP_GLU atom info:{unknownASP_GLUatomInfo}")

        if(potUnknownAtom2):
            for puat2 in potUnknownAtom2:
                unknownASP_GLUatom.append(puat2)                
                unknownASP_GLUatomInfo.append([currASP_GLU[oxygenName2],currASP_GLU.id[1], puat2, puat2.parent.id[1], currASP_GLU[oxygenName2]-puat2])
            if gc.debug:
                print(f"collecting unknownASP_GLU atom info:{unknownASP_GLUatomInfo}")

        #Add the removed atoms back to the search list
        searchASP_GLUatoms.append(currASP_GLU[oxygenName1])
        searchASP_GLUatoms.append(currASP_GLU[oxygenName2])
        if gc.log_file:
            print(f"potUnknownAtom1: {potUnknownAtom1}, potunknownAtom2: {potUnknownAtom2}")

    if(not(unknownASP_GLUatom)):
        unknownASP_GLUatom = []
        unknownASP_GLUatomInfo = []


    return unknownASP_GLUatom,unknownASP_GLUatomInfo


def accomodate_for_ASPs_GLUs(structure, unknownASP_GLU, unknownASP_GLU_atomInfo):

    """
    objective: To account for all ASPs/GLUs in h-bonding distance of each other. If there are over 2 ASPs/GLUs in h-bonding distance-the user will have to take care of that. The user will receive a warning in info file, log_file file and debug file.
    input:-structure: the structure in concern
          -unknownASP_GLU: the unknown of concern
          -unknownASP_GLUatomInfo: additional info including:-atom name, atom id, atom parent id, distance of Oxygen 1 of concern to oxygen in the h-bonding distance.


    output:-structure: updated structures with relevant unknowns marked
           -uniqueUnknownIDs: unknownIDs of ASPs/GLUs that are in h-bonding distance of each other
           -over2ASP_GLUs=1 if there are more than 2 APSs/GLUs in h-bonding distance to each other
    """
    


    fInfoName = get_info_file_name()

    unASP_GLU_atomInfoArr = np.array(unknownASP_GLU_atomInfo)
    allUnknownIDs = unASP_GLU_atomInfoArr[:,1]
    uniqueUnknownIDs = set(allUnknownIDs)
    over2ASP_GLUs  = 0

    ASP_GLU_res = []
    [ASP_GLU_res.append(a.parent) for a in unknownASP_GLU]
    ASP_GLU_res_unique = set(ASP_GLU_res)

    if(len(uniqueUnknownIDs) > 2):
        if gc.log_file:
            print(f"WARNING: There are more than 2 ASPs/GLUs for:\n {ASP_GLU_res_unique} with \n "
                  f"{uniqueUnknownIDs} that are in bonding distance to each other. \n Additional Info :{unASP_GLU_atomInfoArr}\n")
    
        with open(fInfoName, "a") as fInfo:
            fInfo.write(f"WARNING: There are more than 2 ASPs/GLUs for:\n {ASP_GLU_res_unique} with \n {uniqueUnknownIDs} that are in bonding distance to each other. \n Additional Info :{unASP_GLU_atomInfoArr}\n")
            fInfo.flush()

        over2ASP_GLUs = 1
    else:
        for countASP_GLU in range(len(unknownASP_GLU)):
            resID = unknownASP_GLU[countASP_GLU].parent.id
            chainID = unknownASP_GLU[countASP_GLU].parent.parent.id 
            modelID = unknownASP_GLU[countASP_GLU].parent.parent.parent.id
            
            structure[modelID][chainID][resID].isKnown = 0

    if gc.log_file:
        print(f"###################################### ")
        print(f"###################################### ")
        print(
            f"Got unknown ASPs/GLUs: {ASP_GLU_res} with atoms {unknownASP_GLU}.\n And relevant info: {unknownASP_GLU_atomInfo}")

        print(f"Got unknown atoms {unknownASP_GLU}.\n And relevant info:\n {unknownASP_GLU_atomInfo}")

    
    with open(fInfoName, "a") as fInfo:
        fInfo.write(f"###################################### \n")
        fInfo.write(f"###################################### \n")
        fInfo.write(f"Got unknown ASPs/GLUs: {ASP_GLU_res} with atoms {unknownASP_GLU}.\n And relevant info: {unknownASP_GLU_atomInfo}")
        fInfo.write(f"Got unknown atoms {unknownASP_GLU}.\n And relevant info:\n {unknownASP_GLU_atomInfo}")
        fInfo.flush()


    return structure,uniqueUnknownIDs, over2ASP_GLUs


def get_all_residues(structure,  donotIncludeRes = None):
     
    """
    objective: To get all the aminoacid residues in a given structure, model and chain
    I/P: -Structure:structure in concern
         -donotIncludeRes: provide a name of a residue you want left out
    O/P: Gets all Active atoms (that are either acceptors/donors/both for valid 
            amino acids (i.e not hetero residue/water) in the chain

    """
            
    allRes = []
    
    for residue in structure.get_residues():
        r = mra.my_residue(residue)
        if(r.is_valid_amino_acid() and ( residue != donotIncludeRes) and (residue.resname in mc.validResnames)):
            allRes.append(residue)

    return allRes


def get_donor_acceptor_list(structure, aaType = 'ALL'):

     
    """
    Note: 1. This may not necessarily have known list of donor/acceptors!
          2. This is because ASN/GLN may have behaviors but still unknown orientation unlike his
            
    objective: Gets all Active atoms (that are either acceptors/donors/both for valid 
            amino acids (i.e not hetero residue/water) in the chain for a particular behavior.
            -spits out only one list!

    I/P: Structure:structure in concern
         -aaType: define the type of behavior of the atoms required:
                 -DONOR: donor behavior 
                 -ACCEPTOR: acceptor behavior
                 -TBD : only TBD
                 -DONOR_ACCEPTOR_BOTH: donor/acceptor/both 
                 -DONOR_ACCEPTOR_BOTH_TBD: donor/acceptor/both/TBD

    O/P:neededList- requested list of atoms
    """

    neededList = [] 

    for residue in structure.get_residues():
        r = mra.my_residue(residue)
        if r.is_valid_amino_acid():
            for atom in residue:
                a = mra.my_atom(atom)
                abehav = a.get_behavior()[0] 
                if(aaType == 'DONOR'): 
                    if(abehav == 'do'):neededList.append(atom)
                if(aaType == 'ACCEPTOR'): 
                    if(abehav == 'ac'):neededList.append(atom)
                if(aaType == 'TBD'): 
                    if(abehav == 'TBD'):neededList.append(atom)
                if(aaType == 'DONOR_ACCEPTOR_BOTH'): 
                    if(abehav == 'do' or abehav == 'ac' or abehav == 'bo'):neededList.append(atom)
                if(aaType == 'DONOR_ACCEPTOR_BOTH_TBD'): 
                    if(abehav == 'do' or abehav == 'ac' or abehav == 'bo' or abehav == 'TBD'):neededList.append(atom)

    
    return neededList




def get_known_donor_acceptor_list_for_one_atom(structure, at, aaType = 'ALL'):

 
    """
    objective:  get list of known donors/Acceptors/BOTH/ALL/TBD with respect to one atom or parent atoms are included as known!
                -treats parent atom as known and backbone atoms as known!
    I/P:  Structure:structure in concern
         -ModelID:Model in concern
         -chainID: chain in concern
         -aaType: define the type of behavior of the atoms required:
                 -DONOR: donor behavior 
                 -ACCEPTOR: acceptor behavior
                 -DONOR_ACCEPTOR:  donor/acceptor
                 -DONOR_ACCEPTOR_BOTH: donor/acceptor/both 
                 -DONOR_ACCEPTOR_BOTH_TBD: donor/acceptor/both/TBD

    O/P:neededList- requested list of atoms
    """

    neededList = [] 
       
    for residue in structure.get_residues():
        r = mra.my_residue(residue)
        if r.is_valid_amino_acid():
            for atom in residue:
                myACurr = mra.my_atom(atom)
                #checking if the residue is know, or part of backbone or atom is part of parent residue
                cond = ((residue.isKnown == 1) or (myACurr.is_backbone()==1) or (atom in at.parent.child_list))

                if(cond == True):
                    a = mra.my_atom(atom)
                    abehav = a.get_behavior()[0] 
                    if(aaType == 'DONOR'): 
                        if(abehav == 'do'):neededList.append(atom)
                    if(aaType == 'ACCEPTOR'): 
                        if(abehav == 'ac'):neededList.append(atom)
                    if(aaType == 'DONOR_ACCEPTOR'): 
                        if(abehav == 'do' or abehav == 'ac'):neededList.append(atom)
                    if(aaType == 'DONOR_ACCEPTOR_BOTH'): 
                        if(abehav == 'do' or abehav == 'ac' or abehav == 'bo'):neededList.append(atom)
                    if(aaType == 'DONOR_ACCEPTOR_BOTH_TBD'): 
                        if(abehav == 'do' or abehav == 'ac' or abehav == 'bo' or abehav == 'TBD'):neededList.append(atom)
 
    return neededList


def get_unknown_donor_acceptor_list_for_one_atom(structure, at, aaType = 'ALL'):

    """ 
    objective:  get list of unknown donors/Acceptors/BOTH/ALL/TBD
    I/P:  Structure:structure in concern
         -aaType: define the type of behavior of the atoms required:
                 -DONOR: donor behavior 
                 -ACCEPTOR: acceptor behavior
                 -DONOR_ACCEPTOR_BOTH: donor/acceptor/both 
                 -DONOR_ACCEPTOR_BOTH_TBD: donor/acceptor/both/TBD

    O/P: Gets all Active atoms (that are either acceptors/donors/both for valid 
            amino acids (i.e not hetero residue/water) in the chain

    """

    neededList = [] 


    for residue in structure.get_residues():
        r = mra.my_residue(residue)
        if r.is_valid_amino_acid():
            for atom in residue:
                a = mra.my_atom(atom)
                abehav = a.get_behavior()[0] 
                if(residue.isKnown == 0 and a.is_backbone()==0 and (atom not in at.parent.child_list)):
                    if(aaType == 'DONOR'): 
                        if(abehav == 'do'):neededList.append(atom)
                    if(aaType == 'ACCEPTOR'): 
                        if(abehav == 'ac'):neededList.append(atom)
                    if(aaType == 'DONOR_ACCEPTOR_BOTH'): 
                        if(abehav == 'do' or abehav == 'ac' or abehav == 'bo'):neededList.append(atom)
                    if(aaType == 'DONOR_ACCEPTOR_BOTH_TBD'): 
                        if(abehav == 'do' or abehav == 'ac' or abehav == 'bo' or abehav == 'TBD'):neededList.append(atom)

    
    return neededList


def get_unknown_residue_list(structure):

    """
    objective: Get all the residues of concern or all the unknown residues
    I/P: Structure:structure in concern
    O/P: Gets all unknown residues for a given structure, modelID and chainID
    """

    allResUnknown = []

    for residue in structure.get_residues():
        r = mra.my_residue(residue)
        if(r.is_valid_amino_acid() ==1):
            if(residue.isKnown == 0):
                allResUnknown.append(residue)
    return allResUnknown



def get_centroid(structure):

    """
        Objective: Gives the centroid of atoms present in the chain, including waters/small molecules and all chains!!
        I/P: Structure:structure in concern
        O/P: xc:average value of x coordinates of all atoms present in the structure
             yc:average value of y coordinates of all atoms present in the structure
             zc:average value of z coordinates of all atoms present in the structure
        
    """

    numAtoms = 0
    sumXYZ = np.zeros(3)
    count = 0
    for atom in structure.get_atoms():
            count+=1
            a = mra.my_atom(atom)
            S = np.vstack([a.get_coord(), sumXYZ])
            sumXYZ =  np.sum(S, axis = 0)

    numAtoms = count + numAtoms
    xc, yc, zc = sumXYZ/numAtoms

    if gc.debug:
        print(f' \n Centroid (sum all coords/N) is: Xc: {xc}, Yc: {yc}, Zc: {zc}  and Total atoms present: {numAtoms} \n')

    return xc, yc, zc


def normalize_atom_coords(structure):
    """
     objective: Normalize atom coordinates by setting centroid to origin.
        This is done by computing centroid of original coordinates
        and then removing the original coord centroid: (Xc, Yc, Zc) from
        each atom coord.
     I/P: Structure:structure in concern
     O/P: no explicit output-but resetting the structure atom values by removing centroid and thus getting the new centroid as origin
    """
# Compute the centroid which will be used to normalize the coordinates 
    Xc, Yc, Zc = get_centroid(structure)


    for atom in structure.get_atoms():
        atom.set_coord(atom.get_coord()-(Xc, Yc, Zc) )


def set_atom_coords_in_original_frame(structure,xc_orig,yc_orig,zc_orig):
    """
     objective: Revert from normalized atom coordinates by setting centroid to original centroid.
        This is done by adding the original centroid to new coordinated
     I/P: Structure:structure in concern
     O/P: no explicit output-but resetting the structure atom values by adding centroid value of original structure and thus getting the new centroid as that of the original structured
    """

    for atom in structure.get_atoms():
        atom.set_coord(atom.get_coord()+(xc_orig, yc_orig, zc_orig) )

def create_list_of_atoms_of_residue_without_lonepair(res):
    """
    objective: create list of atoms of a residue where lone pair is ignored
    input:res-the residue for which atom list is needed without the lone pair
    output:list of atoms without lone pairs
    """
    listOfAtoms = []
    for atom in res:
        if(atom.id == 'LP1' or atom.id == 'LP2' or atom.id =='LP3' or atom.id =='LP4' or atom.id =='LP5' or atom.id =='LP6'):
            continue
        else:   
            listOfAtoms.append(atom)    

    return listOfAtoms


def detect_clash_for_atoms_of_one_residue(structure, res2check, withinStructClashDist = mc.btwResClashDist):
    """
    objective: To detect if there is a clash with respect to one particular residue in a given structure
    input:Structure:structure in concern
            -res2check: residue with respect to which we are computing the clash
    output: If there is a clash details of the clash is spit to the screen
    """

    ########################
    ###create the residue search list 
    resSearchList  = get_all_residues(structure, donotIncludeRes = res2check)
    ##create atom search list based on the residue
    searchListAtoms = []
    for res in resSearchList:
        searchListAtoms.append(create_list_of_atoms_of_residue_without_lonepair(res))

    searchListAtoms =  [atom for atomlist in searchListAtoms for atom in atomlist]

#   ###use the nbd search and create an object ns.
    ns = Bio.PDB.NeighborSearch(searchListAtoms)
    
    #iterate through the list of atoms of the input residue to look for clash
    atoms2check = res2check.child_list
    for atom in atoms2check:
        #skip if it is a lone pair-as lone pair is pseudo atom
        if(atom.id == 'LP1' or atom.id == 'LP2' or atom.id =='LP3' or atom.id =='LP4' or atom.id =='LP5' or atom.id =='LP6'):
            continue
        else:    
            potClashAtom = ns.search(atom.coord,withinStructClashDist)
            if(potClashAtom and gc.log_file):
                print(f"potential clash atom = {potClashAtom}")
                for pcAtom in potClashAtom:
                   print(f"original atom: {atom}, its coord: {atom.coord}, its parent: {atom.parent}  orig atm parent is rotamer: {atom.parent.isRotamer},")
                   print(f"pcAtom:{pcAtom}, coord:{pcAtom.coord} and pc atm parent: {pcAtom.parent}, pc atm (parent "
                         f"is rotamer): {pcAtom.parent.isRotamer}")
                   print(f"and distance is: {np.linalg.norm(np.float32(atom.coord) - np.float32(pcAtom.coord))} \n")

def detect_clash_within_structure(structure):

    """
        objective: To find clash within the entire structure by going through all its residues
        input:Structure:structure in concern
        output: it will print out details of clash on log file if a clash is found
    """

    allRes  = get_all_residues(structure, donotIncludeRes = None)
    for res in allRes:
        detect_clash_for_atoms_of_one_residue(structure, res, withinStructClashDist = mc.btwResClashDist)

    if gc.log_file:
        print("Looked for clash within the structure\n")

def detect_clash_within_residue(res2check):

    """
    objective: To detect clash within a given residue
    input: res2check: residue considered 
    output: prints to log file if there is a clash, along with details of clash atoms.
    """

    for atom2check in res2check:
        ##Creating a search list
        if(atom2check.id == 'LP1' or atom2check.id == 'LP2' or atom2check.id =='LP3' or atom2check.id =='LP4' or atom2check.id =='LP5' or atom2check.id =='LP6'): continue
        searchListAtoms = create_list_of_atoms_of_residue_without_lonepair(res2check)
        searchListAtoms.remove(atom2check)
        ####use the nbd search and create an object ns.
        ns = Bio.PDB.NeighborSearch(searchListAtoms)
        potClashAtom = ns.search(atom2check.coord, mc.withinResClashDist)

        if(potClashAtom and gc.log_file):
            print(f"potential clash atom = {potClashAtom}\n")
            for pcAtom in potClashAtom:
                print(f"original atom: {atom2check}, its parent: {atom2check.parent}, pcAtom:{pcAtom} and parent: {pcAtom.parent} and distance is: {np.linalg.norm(np.float32(atom2check.coord) - np.float32(pcAtom.coord))}\n")


    

def detect_clash_within_residue_for_all_residues(structure):
    """
    objective: detect clash within a residue(between atoms of the residue) for all residue in a give structure
    input: structure:-the structure that contains residues that needs to be checked.
    output: prints out clash details to log file.
    """

    for res in structure.get_residues():
        r = mra.my_residue(res)
        if r.is_valid_amino_acid():
            detect_clash_within_residue(res)

    if gc.log_file:
        print("Looked for clash within all residues\n")


def setup_structure(protID, outFolder = '.', fName = None):
    
    """
    objective: create a parser for PDB file as a structure. This allows in accessing the models, chains, residues and atoms in a hierarchical manner.
       input: -protID:protID to consider
              -outFolder: destination where data of structures lies.
       output: -structure is a class data type that can be used to easily access residues/atoms

    """
    if(fName == None): fName = protID
    protPDBfile = outFolder + '/' + fName + '.pdb'
    #Create a parser object:
    parser = PDBParser()
    #This object has an attribute get structure parser.get structure returns an object of type structure
    structure = parser.get_structure(protID, protPDBfile) 
    return structure

def change_HIDEP_ASH_GLH(structure):
    """
        objective: If an HIE/HIP/HID is found, switch its name to HIS. 
                   If ASH is found, its name is changed to ASP.
                   IF GLH is found, its name is changed to GLU.
        input: original structure which may have HIE/HID/HIP
        output: structure: structure with only HIS
    """

    for res in structure.get_residues():
        #If residue name is HIE/HID/HIP-then rename it to HIS
        if(res.resname == 'HIE' or res.resname == 'HID' or res.resname == 'HIP'):
            structure[res.parent.parent.id][res.parent.id][res.id].resname = 'HIS'
            if gc.debug:
                print(f"setting res: {res} on chain: {res.parent} and model :{res.parent.parent} as HIS \n")

        #If residue name is ASH-then rename it to ASP
        if(res.resname == 'ASH'):
            structure[res.parent.parent.id][res.parent.id][res.id].resname = 'ASP'
            if gc.debug:
                print(f"setting res: {res} on chain: {res.parent} and model :{res.parent.parent} as ASP \n")

        #If residue name is GLH-then rename it to GLU
        if(res.resname == 'GLH'):
            structure[res.parent.parent.id][res.parent.id][res.id].resname = 'GLU'
            if gc.debug:
                print(f"setting res: {res} on chain: {res.parent} and model :{res.parent.parent} as GLU \n")


    return structure


def set_initial_residue_side_chain_hydrogen_unknown(structure):

    """
    objective: To create the initial set up of defining custom-unknown residues.
    I/P: structure
    O/P: setting SER/THR/TYR into unknown residues, so that the side chain hydrogen is placed on the fly for a given iteration.
    """


    for res in structure.get_residues():
         if res.resname in ['SER', 'THR', 'LYS', 'LYN', 'TYR']:
             res.isSCHknown = 0
             if gc.debug:
                print(f"Setting {res} with chain: {res.parent} as unknown side chain hydrogen since that hydrogen position is not fixed. We need to optimize and find best position. \n")
         else:
             res.isSCHknown = 1


def set_initial_known_residues_and_rotamers(structure):

    """
    objective: To create the initial set up of defining known and rotamer. All residues are not rotamer initially.
    I/P: structure
    O/P: setting knowns and rotamers
    """
    
    for res in structure.get_residues():
         myRes = mra.my_residue(res)
         res.isRotamer = 0
         if(myRes.is_residue_of_concern()==1):
            res.isKnown = 0
            if gc.debug:
                print(f"Setting {res} with chain: {res.parent} as unknown\n")

         else:
            res.isKnown=1


def remove_added_hydrogens(structure):
    """
    objective: Removing hydrogen from all the residues in a given structure
    input:-structure: the structure to consider-where in all residues need to have no hydrogen
    output: updated structure with removed hydrogens
    """

    ARG_H = ['H','HE','HH11','HH12', 'HH21', 'HH22']
    ASH_H = ['H', 'HD2']
    ASP_H = ['H']
    ASN_H = ['H','HD21', 'HD22']
    GLU_H = ['H']
    GLN_H = ['H','HE21','HE22']
    GLH_H = ['H','HE2']
    

    HIS_H = ['H']
    HIE_H = ['H','HE2']
    HID_H = ['H','HD1']
    HIP_H = ['H','HD1','HE2']

    LYS_H = ['H','HZ1','HZ2','HZ3']
    PRO_H = []
    
    SER_H = ['H','HG']
    THR_H = ['H','HG1']
    TYR_H = ['H','HH']
    TRP_H = ['H','HE1']

    ALA_H = ['H']
    CYS_H = ['H']
    GLY_H = ['H']
    ILE_H = ['H']
    LEU_H = ['H']

    MET_H = ['H']
    PHE_H = ['H']
    VAL_H = ['H']

    NME_H = ['H']

    removeHdict = {'ARG':ARG_H, 'ASH':ASH_H, 'ASP':ASP_H, 'ASN':ASN_H,'GLH':GLH_H,'GLU':GLU_H, 'GLN':GLN_H, 
                    'HIS':HIS_H,'HIE':HIE_H, 'HID':HID_H, 'HIP':HIP_H, 'LYS':LYS_H, 'LYN': LYS_H, 'PRO':PRO_H,
                    'SER':SER_H, 'THR':THR_H, 'TYR':TYR_H, 'TRP':TRP_H,
                    'ALA':ALA_H, 'CYS':CYS_H, 'CYX':CYS_H, 'CYM':CYS_H, 'GLY':GLY_H, 'ILE':ILE_H, 'LEU': LEU_H,
                    'MET': MET_H, 'PHE':PHE_H,'VAL':VAL_H, 'NME':NME_H}

    for res in structure.get_residues():
        myRes = mra.my_residue(res)
        if(myRes.is_valid_amino_acid()):
            if(res.resname == 'ACE'): 
                if gc.log_file:
                    print(f"Cannot remove Hydrogen as residue name is: {res.resname}\n")
                continue
            for rmh in removeHdict[res.resname]:
                try:
                    res.detach_child(rmh)
                except KeyError:
                    if gc.log_file:
                        print(f"No Hydrogen were present for: {res.resname} with {res.id[1]} and chain: {res.parent}")
    return structure




def remove_all_hydrogens_from_all_amino_acids(structure):

    """
    objective: Removing hydrogen from all the residues in a given structure
    input:-structure: the structure to consider-where in all residues need to have no hydrogen
    output: updated structure with removed hydrogen
    """

    struct = copy.deepcopy(structure)

    count = 0
    for atom in struct.get_atoms():
        count +=1
        res = atom.parent
        myRes = mra.my_residue(res)
        if(myRes.is_valid_amino_acid() and atom.element == 'H'):
            modelID = atom.parent.parent.parent.id
            chainID = atom.parent.parent.id  
            res2detachID = res.id
            res2detach = structure[modelID][chainID][res2detachID]
            res2detach.detach_child(atom.id)
            
    return structure

def remove_lonepair(structure):
    """
    objective: Removing lone pair atoms from all the residues in a given structure
    input:-structure: the structure to consider-where in all residues need to have no lone pair atoms 
    output: updated structure with removed hydrogen
    """
    ACE_LP = ['LP1','LP2']
    ARG_LP = ['LP1','LP2']
    ASH_LP = ['LP1','LP2','LP3','LP4', 'LP5']
    ASP_LP = ['LP1','LP2','LP3','LP4','LP5','LP6']
    ASN_LP = ['LP1','LP2','LP3','LP4']
    GLU_LP = ['LP1','LP2','LP3','LP4','LP5','LP6']
    GLN_LP = ['LP1','LP2','LP3','LP4' ]
    GLH_LP = ['LP1','LP2','LP3','LP4', 'LP5']
    

    HIS_LP = ['LP1', 'LP2']
    HIE_LP = ['LP1','LP2','LP3']
    HID_LP = ['LP1','LP2','LP3','LP3']
    HIP_LP = ['LP1','LP2']

    LYS_LP = ['LP1','LP2']
    PRO_LP = ['LP1','LP2']
    
    SER_LP = ['LP1','LP2','LP3', 'LP4']
    THR_LP = ['LP1','LP2','LP3','LP4']
    TYR_LP = ['LP1','LP2','LP3' ]
    TRP_LP = ['LP1','LP2']

    ALA_LP = ['LP1', 'LP2']
    CYS_LP = ['LP1', 'LP2']
    GLY_LP = ['LP1', 'LP2']
    ILE_LP = ['LP1', 'LP2']
    LEU_LP = ['LP1', 'LP2']

    MET_LP = ['LP1', 'LP2']
    PHE_LP = ['LP1', 'LP2']
    VAL_LP = ['LP1', 'LP2']

    removeLPdict = {'ACE':ACE_LP,'ARG':ARG_LP, 'ASH':ASH_LP, 'ASP':ASP_LP, 'ASN':ASN_LP,
                    'GLH':GLH_LP,'GLU':GLU_LP, 'GLN':GLN_LP, 
                    'HIS':HIS_LP,'HIE':HIE_LP, 'HID':HID_LP, 'HIP':HIP_LP, 'LYS':LYS_LP, 'LYN': LYS_LP,  'PRO':PRO_LP,
                    'SER':SER_LP, 'THR':THR_LP, 'TYR':TYR_LP, 'TRP':TRP_LP,
                    'ALA':ALA_LP, 'CYS':CYS_LP, 'CYX':CYS_LP, 'CYM':CYS_LP, 'GLY':GLY_LP, 'ILE':ILE_LP , 'LEU': LEU_LP,
                    'MET': MET_LP, 'PHE':PHE_LP,'VAL':VAL_LP}

    for res in structure.get_residues():
        myRes = mra.my_residue(res)
        if(myRes.is_valid_amino_acid()):
            
            if(res.resname == 'NME'):
                if gc.log_file:
                    print(f"Cannot remove LP as residue name is: {res.resname} has no LP\n")
                continue

            for lp in removeLPdict[res.resname]:
                try:
                    res.detach_child(lp)
                except KeyError:
                    if gc.log_file:
                        print(f"No LP were present for: {res.resname} with {res.id[1]} and chain: {res.parent}")
    return structure

def write_to_PDB(structure, fname, removeHLP = False, removeHall = False,set_original_centroid=False):
    """
    objective: write file to .pdb
    input: -structure: structure to write
           -fname: path/fileName to save
           -removeHLP: this flag allows removal of hydrogen and lone pair before writing to file
    output:-file will be written
    """


    if(removeHLP == True):
        structure= remove_added_hydrogens(structure)
        structure= remove_lonepair(structure)
        if gc.debug:
            print("Removed the hydrogen that were added to all the residues\n")
            print("Removed the lonepairs that were added to all the residues\n")

    if(removeHall ==True):
        structure = remove_all_hydrogens_from_all_amino_acids(structure)
        if gc.debug:
            print("removed any hydrogen element present on a residue\n")

    if(set_original_centroid==True):
        set_atom_coords_in_original_frame(structure,mc.xc_orig,mc.yc_orig,mc.zc_orig)

    io = PDBIO()
    io.set_structure(structure)
    if gc.log_file:
        print(f"Writing file at: {fname} ")

    ###if the file is already present-it will overwrite
    io.save(fname)




