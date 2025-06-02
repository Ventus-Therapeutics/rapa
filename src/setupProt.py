import sys
import os
import Bio
import code
import copy

import numpy as np
import myResAtom as mra
import myConstants as mc

from Bio.PDB import *

np.set_printoptions(threshold = sys.maxsize)


def get_allASPs(structure):
    '''
    objective: Get all ASPs in a given structure
    input: -structure: structure you need the ASP for
    output: allASPs: all ASP residues in a list data structure
        
    '''

    allASPs = []

    for res in structure.get_residues():
        if(res.resname == 'ASP'):
            allASPs.append(res)

    return allASPs

def get_allGLUs(structure):

    '''
    objective: Get all GLU in a given structure
    input: -structure: structure you need the GLU for
    output: allGLUs: all GLU residues in a list data structure
        
    '''
    allGLUs = []
    for res in structure.get_residues():
        if(res.resname == 'GLU'):
            allGLUs.append(res)

    return allGLUs

def get_allUnknownASP_ODatoms(structure):
    
    '''
    objective: to get all OD1 and OD2 atoms of ASP 
    input:-structure: the protein structure to consider
    output:-OD1OD2_ASPatoms: list of atoms :OD1 and OD2 of all ASPs in a structure
    '''

    allASPs = get_allASPs(structure)
    OD1OD2_ASPatoms = []

    ##Create allASP atom list for searching nbd!
    for res in allASPs:
        for at in res:
            if(at.id == 'OD1' or at.id =='OD2'):
                OD1OD2_ASPatoms.append(at)
    
    return OD1OD2_ASPatoms

def get_allUnknownGLU_OEatoms(structure):

    '''
    objective: to get all OE1 and OE2 atoms of GLU
    input:-structure: the protein structure to consider
    output:-OE1OE2_GLUatoms: list of atoms :OE1 and OE2 of all GLUs in a structure
    '''


    allGLUs = get_allGLUs(structure)
    OE1OE2_GLUatoms = []

    ##Create allASP atom list for searching nbd!
    for res in allGLUs:
        for at in res:
            if(at.id == 'OE1' or at.id =='OE2'):
                OE1OE2_GLUatoms.append(at)
    
    return OE1OE2_GLUatoms



def get_allUnknownASP_GLU(structure, resName = 'ASP', debug=0):

    '''
    objective: Get all the unknowns ASP or GLUs
    input:-structure: the structure in concern
          -resName:Name of the unknownn residue that needs to be extracted

    output:-unknownASP_GLUatom: list of unknown atoms
           -unknownASP_GLUatomInfo: additional info including:-atom name, atom id, atom parent id, distance of Oxygen 1 of concern to oxygen in the hbonding diatnce.
    '''

    if(debug ==1):
        startDebugFile(__name__, sys._getframe().f_code.co_name)
        fDebugName = get_debugFileName(mc.protID)
        fd = open(fDebugName, "a")
        fd.write(f"searching for all Unknown res: {resName}\n")    
        fd.flush()




    fLogName = get_logFileName(structure.id)
    fLog = open(fLogName, "a")
    #Get all the atoms that needs to be searched for an unknown ASP/GLU residue. Also get a list of all ASPs/GLUs to search
    if(resName == 'ASP'):
        fLog.write("\nChecking for unknown ASPs\n")
        fLog.flush()
        searchASP_GLUatoms = get_allUnknownASP_ODatoms(structure)
        allASPs_GLUs = get_allASPs(structure)
        oxygenName1 = 'OD1'
        oxygenName2 = 'OD2'

    else:
        fLog.write("\nChecking for unknown GLUs\n")
        fLog.flush()
        searchASP_GLUatoms = get_allUnknownGLU_OEatoms(structure)
        allASPs_GLUs = get_allGLUs(structure)
        oxygenName1 = 'OE1'
        oxygenName2 = 'OE2'

    if(debug ==1):

        for atom in searchASP_GLUatoms:
            fd.write("search atom: {atom} of {atom.parent} of chain: {atom.parent.parent}\n")
            fd.flush()
        for res in allASPs_GLUs:
            fd.write("all ASP/GLU residues include: {res} of chain {res.parent}\n")
            fd.flush()


    #create a list of atom to populate upon hitting an unknown Res with OD/OE atoms
    unknownASP_GLUatom = []
    unknownASP_GLUatomInfo = []
    
    for currASP_GLU in allASPs_GLUs:
        #remove side chain oxygen atoms of self residue so we dont count it within hbonding distance
        searchASP_GLUatoms.remove(currASP_GLU[oxygenName1])
        searchASP_GLUatoms.remove(currASP_GLU[oxygenName2])
        #create the search list consisting of all unknown ASPs/GLUs
        ns = Bio.PDB.NeighborSearch(searchASP_GLUatoms)
    
        #use the search object/list  with respect to the two oxygen side chain atoms(OD1/OD2 or OE1/OE2) of the given ASP/GLU
        potUnknownAtom1 = ns.search(currASP_GLU[oxygenName1].coord,4.0)
        potUnknownAtom2 = ns.search(currASP_GLU[oxygenName2].coord,4.0)
        
        #if any of the unknown add it to the unknownASP_GLUatom list and collect relevant info
        if(potUnknownAtom1):
            for puat in potUnknownAtom1: 
                unknownASP_GLUatom.append(puat)
                unknownASP_GLUatomInfo.append([currASP_GLU[oxygenName1],currASP_GLU.id[1], puat, puat.parent.id[1], currASP_GLU[oxygenName1]-puat])
            if(debug==1):
                fd.write("collecting unknownASP_GLU atom info:{unknownASP_GLUatomInfo}")
                fd.flush()

        if(potUnknownAtom2):
            for puat2 in potUnknownAtom2:
                unknownASP_GLUatom.append(puat2)                
                unknownASP_GLUatomInfo.append([currASP_GLU[oxygenName2],currASP_GLU.id[1], puat2, puat2.parent.id[1], currASP_GLU[oxygenName2]-puat2])
            if(debug==1):
                fd.write("collecting unknownASP_GLU atom info:{unknownASP_GLUatomInfo}")
                fd.flush()

        #Add the removed atoms back to the search list-This step is kinda redundant as we are not using the search list anywhere else 
        searchASP_GLUatoms.append(currASP_GLU[oxygenName1])
        searchASP_GLUatoms.append(currASP_GLU[oxygenName2])
        fLog.write(f"potUnknownAtom1: {potUnknownAtom1}, potunknownAtom2: {potUnknownAtom2}\n")
        fLog.flush()

    if(not(unknownASP_GLUatom)):
        unknownASP_GLUatom = []
        unknownASP_GLUatomInfo = []

    if(debug ==1):
        endDebugFile(__name__,sys._getframe().f_code.co_name, fd)

    fLog.close()
    return unknownASP_GLUatom,unknownASP_GLUatomInfo


def accomodateForASP_GLUs(structure, unknownASP_GLU, unknownASP_GLU_atomInfo):
            
    fLogName = get_logFileName(structure.id)
    f = open(fLogName, "a")

    unASP_GLU_atomInfoArr = np.array(unknownASP_GLU_atomInfo)
    allUnknownIDs = unASP_GLU_atomInfoArr[:,1]
    uniqueUnknownIDs = set(allUnknownIDs)
    over2ASP_GLUs  = 0

    ASP_GLU_res = []
    [ASP_GLU_res.append(a.parent) for a in unknownASP_GLU]
    ASP_GLU_res_unique = set(ASP_GLU_res)

    if(len(uniqueUnknownIDs) > 2):
        f.write(f"\n WARNING: There are more than 2 ASPs/GLUs for:\n {ASP_GLU_res_unique} with \n {uniqueUnknownIDs} that are in bonding distance to each other. \n Additional Info :{unASP_GLU_atomInfoArr}\n")
        over2ASP_GLUs = 1
    else:
        for countASP_GLU in range(len(unknownASP_GLU)):
            resID = unknownASP_GLU[countASP_GLU].parent.id
            chainID = unknownASP_GLU[countASP_GLU].parent.parent.id 
            modelID = unknownASP_GLU[countASP_GLU].parent.parent.parent.id
            
            structure[modelID][chainID][resID].isKnown = 0

    f.write(f"###################################### \n")
    f.write(f"###################################### \n")
    f.write(f"\n\nGot unknown ASPs/GLUs: {ASP_GLU_res} with atoms {unknownASP_GLU}.\n And relevant info: {unknownASP_GLU_atomInfo}\n")

    f.write(f"\n\nGot unknown atoms {unknownASP_GLU}.\n And relevant info:\n {unknownASP_GLU_atomInfo}\n")
    f.flush()
    f.close()

    return structure,uniqueUnknownIDs, over2ASP_GLUs





def get_allResidues(structure,  donotIncludeRes = None ):
    ''' 
    objective: To get all the aminoacid residues in a given structure, model and chain
    I/P: -Structure:structure in concern
         -ModelID:Model in concern
         -chainID: chain in concern
         -donotIncludeRes: provide a name of a residue you want left out
    O/P: Gets all Active atoms (that are either acceptors/donors/both for valid 
            amino acids (i.e not hetero residue/water) in the chain

            '''
    allRes = []
    
    for residue in structure.get_residues():
        r = mra.myResidue(residue)
        if(r.isValidAminoAcid() and ( residue != donotIncludeRes) and (residue.resname in mc.validResnames)):
            allRes.append(residue)

    return allRes



def get_activeAtoms_allInfo(structure):

    ''' 
        objective: gets all Active atoms (that are either acceptors/donors/both for valid 
            amino acids (i.e not hetero residue/water) in the chain
        I/P: -Structure:structure in concern
         -ModelID:Model in concern
         -chainID: chain in concern

        O/P:-allActiveAtoms: list of all active atoms and its relevant info including:
            parent residue id, parent residue name, atom name, atom behavior, atom coords          
            '''

    allActiveAtoms = []

    for residue in structure.get_residues():
        r = mra.myResidue(residue)
        if r.isValidAminoAcid():
            for atom in residue: 
                a = mra.myAtom(atom)                
                if a.get_behavior()[0] != 'XX':
                    allActiveAtoms.append([r.get_id()[1], r.get_name(), a.get_name(), a.get_behavior().IDval, a.get_behavior().abbrev, a.get_coord()[0], a.get_coord()[1], a.get_coord()[2]])
    
    return allActiveAtoms




def get_allActiveAtomsInfo(structure):
    ''' 
    objective: To get all active atom(acceptor/donor/both) info for valid amino acid in a separate lists for a given structure
    I/P: Structure, ModelID, chainID
    O/P: allActiveAtoms: all active atoms(donor/acceptor/both/TBD)
         allDonors: donor/both/TBD
         allAcceptors: acceptor/both/TBD
         allBoth: list of atoms where atom behave as both-acceptor and donor
         allTBD:-all To be determined(or unknown) residue atoms
         allOnlyDonors:-with only donor behavior
         allOnlyAcceptors:-with only acceptor behavior
         allOnlyDonorsAcceptors: with donors or acceptors
         allDonorsAcceptorsBoth: with donors or acceptors or both

            '''
    allActiveAtoms = []
    allDonors = []
    allAcceptors = []
    allOnlyDonors = []
    allOnlyAcceptors = []
    allOnlyDonorsAcceptors = []
    allDonorsAcceptorsBoth = []


    allTBD = []
    allBoth = []
    
    for residue in structure.get_residues():

        r = mra.myResidue(residue)

        if r.isValidAminoAcid():
            for atom in residue: 
                a = mra.myAtom(atom)
                abehav = a.get_behavior()[0] 
                if( abehav == 'do'):allOnlyDonors.append(atom)
                if( abehav == 'ac'): allOnlyAcceptors.append(atom)
                if( abehav == 'do' or abehav == 'bo' or abehav == 'TBD'):allDonors.append(atom)
                if( abehav == 'ac' or abehav == 'bo' or abehav == 'TBD'): allAcceptors.append(atom)
                if( abehav == 'do' or abehav == 'ac'):allOnlyDonorsAcceptors.append(atom)
                if( abehav == 'do' or abehav == 'ac' or abehav == 'bo'):allDonorsAcceptorsBoth.append(atom)
                if( abehav == 'TBD'): allTBD.append(atom)
                if( abehav == 'bo'): allBoth.append(atom)
                if( abehav != 'XX'): allActiveAtoms.append(atom)
    
    return allActiveAtoms, allDonors, allAcceptors, allBoth, allTBD, allOnlyDonors, allOnlyAcceptors, allOnlyDonorsAcceptors, allDonorsAcceptorsBoth



def get_DonorAcceptorList(structure, aaType = 'ALL'):

    ''' 
    NOTE: 1. THIS MAY NOT NECESSARILY HAVE KNOWN LIST OF DONOR/ACCEPTORS!
          2. THIS IS CUZ ASN/GLN MAY HAVE BEHAVIORS BUT STILL UNKNOWN ORIENTATION UNLIKE HIS
            
    objective: Gets all Active atoms (that are either acceptors/donors/both for valid 
            amino acids (i.e not hetero residue/water) in the chain for a particular behavior.
            -spits out only one list!

    I/P: Structure:structure in concern
         -ModelID:Model in concern
         -chainID: chain in concern
         -aaType: define the type of behavior of the atoms required:
                 -DONOR: donor behavior 
                 -ACCEPTOR: acceptor behavior
                 -TBD : only TBD
                 -DONOR_ACCEPTOR_BOTH: donor/acceptor/both 
                 -DONOR_ACCEPTOR_BOTH_TBD: donor/acceptor/both/TBD

    O/P:neededList- requested list of atoms
            '''

    neededList = [] 

    for residue in structure.get_residues():
        r = mra.myResidue(residue)
        if r.isValidAminoAcid():
            for atom in residue:
                a = mra.myAtom(atom)
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




def get_knownDonorAcceptorListWRTOneAtom(structure, at, aaType = 'ALL'):

    ''' 
    objective:  get list of KNOWN donors/Acceptors/BOTH/ALL/TBD WRT to one atom or parent atoms are included as known!
                -treats parent atom as known and backbone atoms as knowns!
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

            '''

    neededList = [] 
       
    for residue in structure.get_residues():
        r = mra.myResidue(residue)
        if r.isValidAminoAcid():
            for atom in residue:
                myACurr = mra.myAtom(atom)
                #checking if the residue is know, or part of backbone or atom is part of parent residue
                cond = ((residue.isKnown == 1) or (myACurr.isBackbone()==1) or (atom in at.parent.child_list))

                if(cond == True):
                    a = mra.myAtom(atom)
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


def get_unknownDonorAcceptorListWRTOneAtom(structure, at, aaType = 'ALL'):
    ''' 
    NOTE: Currently using it as a hack to check for unknown atoms that DO NOT belong to the parent atoms/backbone, and is unknown
    objective:  get list of UNKNOWN donors/Acceptors/BOTH/ALL/TBD
    I/P:  Structure:structure in concern
         -ModelID:Model in concern
         -chainID: chain in concern
         -aaType: define the type of behavior of the atoms required:
                 -DONOR: donor behavior 
                 -ACCEPTOR: acceptor behavior
                 -DONOR_ACCEPTOR_BOTH: donor/acceptor/both 
                 -DONOR_ACCEPTOR_BOTH_TBD: donor/acceptor/both/TBD

    O/P: Gets all Active atoms (that are either acceptors/donors/both for valid 
            amino acids (i.e not hetero residue/water) in the chain

            '''

    neededList = [] 


    for residue in structure.get_residues():
        r = mra.myResidue(residue)
        if r.isValidAminoAcid():
            for atom in residue:
                a = mra.myAtom(atom)
                abehav = a.get_behavior()[0] 
                if(residue.isKnown == 0 and a.isBackbone()==0 and (atom not in at.parent.child_list)):
                    if(aaType == 'DONOR'): 
                        if(abehav == 'do'):neededList.append(atom)
                    if(aaType == 'ACCEPTOR'): 
                        if(abehav == 'ac'):neededList.append(atom)
                    if(aaType == 'DONOR_ACCEPTOR_BOTH'): 
                        if(abehav == 'do' or abehav == 'ac' or abehav == 'bo'):neededList.append(atom)
                    if(aaType == 'DONOR_ACCEPTOR_BOTH_TBD'): 
                        if(abehav == 'do' or abehav == 'ac' or abehav == 'bo' or abehav == 'TBD'):neededList.append(atom)

    
    return neededList



def get_unknownResList(structure):

    '''
    objective: Get all the residues of concern or all the unknownRes
    I/P: Structure:structure in concern
         -ModelID:Model in concern
         -chainID: chain in concern

    O/P: Gets all unknown residues for a given structure, modelID and chainID
    '''

    allResUnknown = []

    for residue in structure.get_residues():
        r = mra.myResidue(residue)
        if(r.isValidAminoAcid() ==1):
            if(residue.isKnown == 0):
                allResUnknown.append(residue)
    return allResUnknown


def get_allAminoAcidAtoms(structure):

    '''Objective: Gets all atom info for valid amino acids (i.e not hetero residue/water) in the chain
       I/P: Structure:structure in concern
            -ModelID:Model in concern
            -chainID: chain in concern

        O/P: allAtoms-list of all valid aminoacids


    '''
    allAtoms = []

    for residue in structure.get_residues():
        r = mra.myResidue(residue)
        if r.isValidAminoAcid():
            for atom in residue: 
                    allAtoms.append(atom)
    return allAtoms

def get_centroid(structure):

    '''Objective: Gives the centroid of atoms present in the chain, including waters/small molecules and all chains!!
        I/P: Structure:structure in concern
            -ModelID:Model in concern
            -chainID: chain in concern
        O/P: xc:average value of x coordinates of all atoms present in the structure
             yc:average value of y coordinates of all atoms present in the structure
             zc:average value of z coordinates of all atoms present in the structure
        
    '''

    numAtoms = 0
    sumXYZ = np.zeros(3)
    count = 0
    for atom in structure.get_atoms():
            count+=1
            a = mra.myAtom(atom)
            S = np.vstack([a.get_coord(), sumXYZ])
            sumXYZ =  np.sum(S, axis = 0)

    numAtoms = count + numAtoms
    xc, yc, zc = sumXYZ/numAtoms

    append2log(f' \n Centroid (sum all coords/N) is: Xc: {xc}, Yc: {yc}, Zc: {zc}  and Total atoms present: {numAtoms} \n')

    return xc, yc, zc




def normalizeAtomCoords(structure):
    '''
     objective: Normalize atom coords by setting centroid to origin. 
        This is done by computing computing centroid of original coords
        and then removing the orignal coord centroid: (Xc, Yc, Zc) from 
        each atom coord.

     I/P: Structure:structure in concern
            -ModelID:Model in concern
            -chainID: chain in concern

     O/P: no explicit output-but resetting the structure atom values by removing centroid and thus getting the new centroid as origin


        '''
# Compute the centroid which will be used to normalize the coordinates 
    Xc, Yc, Zc = get_centroid(structure)


    for atom in structure.get_atoms():
        atom.set_coord(atom.get_coord()-(Xc, Yc, Zc) )


def createListOfAtomsOfResidueWithoutLP(res):
    '''
    objective: create list of atoms of a residue where lone pair is ignored
    input:res-residue who's atom list is needed without the lone pair
    output:list of atoms without lone pairs
    '''
    listOfAtoms = []
    for atom in res:
        #idv = atom.id
        if(atom.id == 'LP1' or atom.id == 'LP2' or atom.id =='LP3' or atom.id =='LP4' or atom.id =='LP5' or atom.id =='LP6'):
            continue
        else:   
            listOfAtoms.append(atom)    

    return listOfAtoms



def detectClashWrtResidue(structure, res2check, withinStructClashDist = mc.btwResClashDist):
    '''
    objective: To detect if there is a clash with respect to one particular residue in a given structure
    input:Structure:structure in concern
            -ModelID:Model in concern
            -chainID: chain in concern
            -res2check: residue with respect to which we are computing the  clash

    output: If there is a clash details of the clash is spit to the screen
    '''    

    ######get and open log file###############
    fLogName = get_logFileName(structure.id)
    fLog = open(fLogName, "a")
    outputFolder = get_outputFolderName(structure.id)
    ########################
    ###create the residue search list 
    resSearchList  = get_allResidues(structure, donotIncludeRes = res2check)
    ##create atom search list based on the residue
    searchListAtoms = []
    for res in resSearchList:
        searchListAtoms.append(createListOfAtomsOfResidueWithoutLP(res))
    ##Flatten it
    searchListAtoms =  [atom for atomlist in searchListAtoms for atom in atomlist]

#        ###use the nbd search and create an object ns.
    ns = Bio.PDB.NeighborSearch(searchListAtoms)
    
    #iterate through the list of atoms of the input residue to look for clash
    atoms2check = res2check.child_list
    for atom in atoms2check:
        #skip if it is a lone pair-as lone pair is just pseudo atom    
        if(atom.id == 'LP1' or atom.id == 'LP2' or atom.id =='LP3' or atom.id =='LP4' or atom.id =='LP5' or atom.id =='LP6'):
            continue
        else:    
            potClashAtom = ns.search(atom.coord,withinStructClashDist)
            if(potClashAtom):
                fLog.write(f"potential clash atom = {potClashAtom}")
                for pcAtom in potClashAtom:
                    fLog.write(f"orignal atom: {atom}, its coord: {atom.coord}, its parent: {atom.parent}  orig atm parent is rotamer: {atom.parent.isRotamer},\n pcAtom:{pcAtom}, coord:{pcAtom.coord} and pc atm parent: {pcAtom.parent}, pc atm parent is rotamer: {pcAtom.parent.isRotamer}\n and distance is: {np.linalg.norm(np.float32(atom.coord)-np.float32(pcAtom.coord))} \n")
    fLog.close()
##########################################################################################################################

def detectClashWithinStructure(structure,withinStructClashDist = mc.btwResClashDist):

    '''
        objective: To find clash within the entire structure by going through all its residues
        input:Structure:structure in concern
            -ModelID:Model in concern
            -chainID: chain in concern
        output: it will print out details of clash on screen if a clash is found
    '''


    allRes  = get_allResidues(structure, donotIncludeRes = None)
    for res in allRes:
        detectClashWrtResidue(structure, res, withinStructClashDist = mc.btwResClashDist)

   
    fLogName = get_logFileName(structure.id)
    fLog = open(fLogName, "a")
    fLog.write("Looked for clash within the structure\n")
    fLog.close()

def detectClashWithinResidue(res2check, withinResClashDist =mc.withinResClashDist):

    '''
    objective: To detect clash within a given residue
    input: res2check: residue considered 
    output: prints to screen if there is a clash-details on the clash!
    '''
    #########Get Log file and open/append##########
    structure = res2check.parent.parent.parent
    fLogName = get_logFileName(structure.id)
    fLog = open(fLogName, "a")

    for atom2check in res2check:
        ##Creating a search list
        if(atom2check.id == 'LP1' or atom2check.id == 'LP2' or atom2check.id =='LP3' or atom2check.id =='LP4' or atom2check.id =='LP5' or atom2check.id =='LP6'): continue
        searchListAtoms = createListOfAtomsOfResidueWithoutLP(res2check)
        searchListAtoms.remove(atom2check)
        ####use the nbd search and create an object ns.
        ns = Bio.PDB.NeighborSearch(searchListAtoms)

        potClashAtom = ns.search(atom2check.coord, withinResClashDist)
        if(potClashAtom):
            fLog.write(f"potential clash atom = {potClashAtom}\n")
            for pcAtom in potClashAtom:
                fLog.write(f"orignal atom: {atom2check}, its parent: {atom2check.parent}, pcAtom:{pcAtom} and parent: {pcAtom.parent} and distance is: {np.linalg.norm(np.float32(atom2check.coord)-np.float32(pcAtom.coord))}\n")
    fLog.close()

def detectClashWithinResidueForAllResidues(structure, withinResClashDist =mc.withinResClashDist): 
    '''
    objective: detect clash within a residue(between atoms of the residue) for all residue in a give structure
    input: structure:-the structure who's residues needs to be checked. 
    output: prints out clash details to screen
    '''
######get and open log file###############
    fLogName = get_logFileName(structure.id)
    fLog = open(fLogName, "a")

    for res in structure.get_residues():
        r = mra.myResidue(res)
        if r.isValidAminoAcid():
            detectClashWithinResidue(res, withinResClashDist =mc.withinResClashDist)
    fLog.write("Looked for clash within all residues\n")
    fLog.close()
#########################################################################################################################

def setupStructure(protID, outFolder = '.', fName = None, debug=0):
    
    '''objective: Downloading the cif, creating a parser for PDB files and creating the structure  
       input: -protID:protID to consider
              -outFolder: destination where data of structures lies. /PDB/ is internally added to work with pdb files
       output: -structure is a class data type that can be used to easily access residues/atoms

    '''
    if(fName == None): fName = protID
    protPDBfile = outFolder + '/' + fName + '.pdb'
    #Create a parser object:
    parser = PDBParser()
    #This object has an attribute get structure parser.get structure returns an object of type structure
    structure = parser.get_structure(protID, protPDBfile) 
    return structure



def set_initialResSC_hydKnown(structure):

    ''' 
    objective: To create the initial set up of defining knowns and rotamers. All residues are not rotamers initially.
    I/P: structure
    O/P: setting knowns and rotamers
    '''
    for res in structure.get_residues():
         if(res.resname == 'SER' or res.resname == 'THR' or res.resname == 'LYS' or res.resname == 'TYR'):
             res.isSCHknown = 0
         else:
             res.isSCHknown = 1

def set_initialKnownAndRotamers(structure,debug=0):

    ''' 
    objective: To create the initial set up of defining knowns and rotamers. All residues are not rotamers initially.
    I/P: structure
    O/P: setting knowns and rotamers

    '''
    if(debug ==1):
        startDebugFile(__name__, sys._getframe().f_code.co_name)
        fDebugName = get_debugFileName(mc.protID)
        fd = open(fDebugName, "a")
    
    for res in structure.get_residues():
         myRes = mra.myResidue(res)
         res.isRotamer = 0
         if(myRes.isResidueOfConcern()==1):
            res.isKnown = 0
            if(debug==1): 
                fd.write(f"Setting {res} with chain: {res.parent} as unknown\n")
                fd.flush()
         else:
            res.isKnown = 1
    
    if(debug ==1):
        endDebugFile(__name__,sys._getframe().f_code.co_name, fd)

def startDebugFile(modName, funcName):
        
    fDebugName = get_debugFileName(mc.protID)
    fd = open(fDebugName, "a")
    fd.write(f"\n\n###############################################################################\n")    
    fd.write(f"********************Enter module {modName} at function: {sys._getframe().f_code.co_name} ********************\n")    
    fd.flush()

def endDebugFile(modName, funcName, fd):

    fd.write(f"********************Exit module {modName} at function: {funcName}********************\n")    
    fd.write(f"###############################################################################\n\n")    
    fd.flush()
    fd.close()


def remove_H(structure):
    '''
    objective: Removing hydrogen from all the residues in a given structure
    input:-structure: the structure to consider-where in all residues need to have no hydrogen
    output: updated structure with removed hydrogens
    '''

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
                    'HIS':HIS_H,'HIE':HIE_H, 'HID':HID_H, 'HIP':HIP_H, 'LYS':LYS_H, 'PRO':PRO_H,
                    'SER':SER_H, 'THR':THR_H, 'TYR':TYR_H, 'TRP':TRP_H,
                    'ALA':ALA_H, 'CYS':CYS_H, 'GLY':GLY_H, 'ILE':ILE_H, 'LEU': LEU_H, 'MET': MET_H,
                    'PHE':PHE_H,'VAL':VAL_H, 'NME':NME_H}

    for res in structure.get_residues():
        myRes = mra.myResidue(res)
        if(myRes.isValidAminoAcid()):
            if(res.resname == 'ACE'): 
                append2log(f"Cannot remove Hydrogen as residue name is: {res.resname}\n")
                #print(f"Cannot remove Hydrogen as residue name is: {res.resname}")
                continue
            for rmh in removeHdict[res.resname]:
                try:
                    res.detach_child(rmh)
                except KeyError:
                    append2log(f"No Hydrogens were present for: {res.resname} with {res.id[1]} and chain: {res.parent}\n" )
    return structure



def remove_H_all(structure):

    '''
    objective: Removing hydrogen from all the residues in a given structure
    input:-structure: the structure to consider-where in all residues need to have no hydrogen
    output: updated structure with removed hydrogens
    '''

    struct = copy.deepcopy(structure)

    count = 0
    for atom in struct.get_atoms():
        count +=1
        res = atom.parent
        myRes = mra.myResidue(res)
        if(myRes.isValidAminoAcid() and atom.element == 'H'):
            modelID = atom.parent.parent.parent.id
            chainID = atom.parent.parent.id  
            res2detachID = res.id
            res2detach = structure[modelID][chainID][res2detachID]
            res2detach.detach_child(atom.id)
            
    return structure

def remove_LP(structure):
    '''
    objective: Removing lone pair atoms from all the residues in a given structure
    input:-structure: the structure to consider-where in all residues need to have no lone pair atoms 
    output: updated structure with removed hydrogens
    '''
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
                    'HIS':HIS_LP,'HIE':HIE_LP, 'HID':HID_LP, 'HIP':HIP_LP, 'LYS':LYS_LP, 'PRO':PRO_LP,
                    'SER':SER_LP, 'THR':THR_LP, 'TYR':TYR_LP, 'TRP':TRP_LP,
                    'ALA':ALA_LP, 'CYS':CYS_LP, 'GLY':GLY_LP, 'ILE':ILE_LP , 'LEU': LEU_LP, 'MET': MET_LP,
                    'PHE':PHE_LP,'VAL':VAL_LP}

    for res in structure.get_residues():
        myRes = mra.myResidue(res)
        if(myRes.isValidAminoAcid()):
            
            if(res.resname == 'NME'): 
                append2log(f"Cannot remove LP as residue name is: {res.resname} has no LP\n")
                continue

            for lp in removeLPdict[res.resname]:
                try:
                    res.detach_child(lp)
                except KeyError:
                    append2log(f"No LP were present for: {res.resname} with {res.id[1]} and chain: {res.parent} \n" )
    return structure


def write2PDB(structure, fname, removeHLP = False, removeHall = False):
    '''
    objective: write file to .pdb
    input: -structure: structure to write
           -fname: path/fileName to save
           -removeHLP: this flag allows removal of hydrogen and lone pair before writing to file
    output:-file will be written

    '''
    if(removeHLP == True):
        structure= remove_H(structure)
        structure= remove_LP(structure)
    if(removeHall ==True):
        structure = remove_H_all(structure)

    io = PDBIO()
    io.set_structure(structure)
    append2log(f"Writing file at: {fname}  \n")
    ###CHECK if the file is already present-it will overwrite
    io.save(fname)


def get_outputFolderName(structureID, otp_sIDname = False):

    '''
    objective: To return name of the output folder
    I/P:
    -structureID: ID of the structure you are working with
    -otp_sIDname =1 if you want struct ID output name (it is: structID_HLPsp2)
                 =0 if you only want the output folder name

    O/P:
    -outputFolder: name of the output folder
    -sIDName: structure ID name(structureID_HLPsp2) you want to use in the output folder name

    '''

    if(structureID[-6:]=='HLPsp2'):
        sIDName = structureID
    else:
        sIDName = structureID + '_HLPsp2'

    outputFolder = "OUTPUTS_"+sIDName

    if(otp_sIDname ==True):
        return outputFolder, sIDName
    else:
        return outputFolder



def get_logFileName(structureID):
    
    '''
    objective: To get the name of the log file
    Input:
    -structureID: ID of the structure you are working with
    Output:
    -fLogName: name of the log file

    '''    

    outputFolder, sID = get_outputFolderName(structureID, otp_sIDname = True)

    fDest = f'{outputFolder}/'
    checkFolderPresent = os.path.isdir(fDest)    
    if not checkFolderPresent: os.makedirs(fDest)


    fLogName = f"{fDest}"+f"{sID}.log"
    
    return fLogName

def get_infoFileName(structureID):
    
    '''
    objective: To get the name of the info file name
    Input:
    -structureID: ID of the structure you are working with
    Output:
    -fInfoName: name of the info file
    '''    

    outputFolder, sID = get_outputFolderName(structureID, otp_sIDname = True)

    fDest = f'{outputFolder}/'
    checkFolderPresent = os.path.isdir(fDest)    
    if not checkFolderPresent: os.makedirs(fDest)

    fInfoName = f"{fDest}"+f"{sID}.info"
    
    return fInfoName

def get_debugFileName(structureID):
    
    '''
    objective: To get the name of the info file name
    Input:
    -structureID: ID of the structure you are working with
    Output:
    -fDebugName: name of the info file
    '''    

    outputFolder, sID = get_outputFolderName(structureID, otp_sIDname = True)

    fDest = f'{outputFolder}/'
    checkFolderPresent = os.path.isdir(fDest)    
    if not checkFolderPresent: os.makedirs(fDest)

    fDebugName = f"{fDest}"+f"{sID}.debug"
    
    return fDebugName

def get_pdbOutFolder(structureID):
    
    '''
    objective: To get the name of the folder where the output pdbs are stored
    Input:
    -structureID: ID of the structure you are working with
    Output:
    -foPDB: name of the pdb folder
    '''    

    outputFolder, sID = get_outputFolderName(structureID, otp_sIDname = True)

    foPDB = f'{outputFolder}/PDB_out'
    checkFolderPresent = os.path.isdir(foPDB)    
    if not checkFolderPresent: os.makedirs(foPDB)

    
    return foPDB

def append2log(MSG):
    '''
    objective: To output a message to log file with 'protID_HLPsp2.log' 
               Note: use this function only after creating the log file
    Input:-MSG: message to output
    '''
        
    fLogName = get_logFileName(mc.protID)
    fLog = open(fLogName, "a")
    fLog.write(MSG)
    fLog.flush()
    fLog.close()

