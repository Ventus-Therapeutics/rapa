import os
import sys
import csv
import Bio
import numpy as np
import code
import copy

from os.path import exists
from Bio.PDB import *

import pandas as pd
import setupProt as stp
import myResAtom as mra
import myConstants as mc
import hPlacementSP2 as hsp2


np.set_printoptions(threshold = sys.maxsize)

def hbond_energy(r, theta, gamma, attractive=True,
                 D1=5.3690476811615016,D2=27.58319209788471,D3=39.701896317848934,D4=4.854675162065331e-10,D5=0.09999999999999998,
                 r1=-0.07438325810984132,r2=1.400238259764079,r3=0.4857252493470585,r4=-0.8284743134969731,
                 C1=-5.608522269624793,C2=27.0906634067007,C3=40.17918959663446,C4=4.707296378543495e-10,C5=5.355210499992828):

    # Computes the relative energy difference [kcal/mol] for a hydrogen bond
    # Data was fit to a rigid body scan of methanol (donor) and water acceptor
    # Energies calculated at the RHF/6-31G(d,p) level of theory
    # Created by Nathan S. Abraham (nabraham@ventustx.com)

    def fit_1(t, g, f1, f2, f3, f4, f5):
        return f1*np.exp(-(t**2/(2*f2**2) + g**2/(2*f3**2) + f4*t*g)) + f5

    def fit_2(t, g, f1, f2, f3, f4):
        #return (f2 + f1*np.cos(np.radians(t)))**4 + f4*(1 - f3*np.cos(np.radians(t)))*np.cos(np.radians(g))
        return (f2 + f1*np.cos(np.radians(t)))**4 + (f4 + f3*np.cos(np.radians(g)))**4

    def morse(R, t, g):
        U_inf = 5.215735911529536
        D = fit_1(t, g, D1, D2, D3, D4, D5)
        A = 1.45
        rm = fit_2(t, g, r1, r2, r3, r4)
        C = fit_1(t, g, C1, C2, C3, C4, C5)
        return D*(1 - np.exp(-A*(R - rm)))**2 + C - U_inf

    u = morse(r, theta, gamma)
    if -90.0 < gamma < 90.0 and -90.0 < theta < 90.0:
        if attractive:
            if u < 0:
                return u
            elif morse(r, 0.0, 0.0) < 0:
                return 0.0
            else:
                return u
        else:
            step_size = 0.001
            derivative = (morse(r+step_size, theta, gamma) - morse(r-step_size, theta, gamma)) / (2*step_size)
            if derivative < 0.0:
                steps = np.arange(1.5,4.0,0.01)
                alt_u = -1*morse(steps, theta, gamma)
                if max(alt_u) < 0:
                    return 0.0
                else:
                    return max(alt_u)
            else:
                if u > 0 and morse(r, 0.0, 0.0) < 0:
                    return 0
                else:
                    return -1*u
    else:
        return 0.0


def get_closeAtomList(targAtom, givenList, debug=0):

    ''' 
    objective: Get a list of close atom for a target atom
    Input: Target atom which is a myAtom type object.

    O/P: A List of atoms that in the radius of: myConstants.deltaD
         The output is a list of the atom objects.

    '''
    
    if(not givenList): 
        stp.append2log("No search List provided!!-STOP! \n", debug=0)
        os._exit(0)


    ns = Bio.PDB.NeighborSearch(givenList)
    close_atoms = ns.search(targAtom.coord, mc.deltaD) ##distance cut off is defined by mc.deltaD
    

    if(targAtom not in close_atoms):
        close_atoms.append(targAtom)
    #
    return close_atoms


def get_closeAtomDistInfo(atomCurr, close_atoms, debug = 0):
    '''
        objective: To provide a list and info regarding close atoms, close atom parent, close atom behavior and distance 
                    from current atom
        I/P: Current Atom, and the list of close atoms you want information about
        O/P: Distance information in asscending order. The first 
    '''
    targAtom = mra.myAtom(atomCurr)
    distInfo = []

    for atomClose in close_atoms:
       aClose = mra.myAtom(atomClose)
       dist = atomClose-atomCurr
       distInfo.append([atomClose, atomClose.get_parent().id[1],atomClose.get_parent().resname, aClose.get_behavior().abbrev, dist])

    
    #sorting the array according the distance- smallest to largest:
    distInfo.sort(key = lambda i: i[-1])
   
    return distInfo

def get_allCloseAtomInfoForOneAtom(aCurr, givenList, debug =0 ):

    '''
        I/P: 
            -aCurr: current atom
            -searchListName: search list 
            -givenList
        O/P: A close atom matrix has first row as an active atom and its related info in the columns
        All atoms within the DeltaD radius is listed in the next few rows, with the closest
        atom being on row 1. The atoms are listed in closest to farthest order.        
       '''
    close_atoms = get_closeAtomList(aCurr, givenList, debug=debug)

    if np.shape(close_atoms)[0] > 0:
        distInfo = get_closeAtomDistInfo(aCurr, close_atoms, debug=debug)
    else:
        return []

    return distInfo



def get_listOfCloseAtomsForListOfAtoms(listOfInputAtoms,  aaType, debug = 0):
    
    '''
    Objective: To get close atoms for a list of atoms. #useWithCaution
                #This list of close atoms may not contain only known atoms!!
    Input: -listOfInputAtoms: list of input atoms you want close atoms for
           -aatype: type of active atoms for list of close atoms, includes:
                    -DONOR: only donor atoms
                    -ACCEPTOR: only acceptor atoms
                    -DONOR_ACCEPTOR_BOTH: only donor/
                    -DONOR_ACCEPTOR_BOTH_TBD: Donor,acceptor, both and to be determined
                    #NOTE: the atoms may not necessarily be correctly placed-for example ASN may not be set, yet it has
                            donor and acceptors.
    Output:-LOCA: List of Close Atoms
           -dim: Number of Close Atoms including its own atom and all the info provided with it..i.e:
            [atom, atom.parent.id, atom.parent.resname, distance from original atom]

    '''

    #create list of close atoms and dimensions
    LOCA = []
    dim = []

    #iterate over all atoms in the list of input atoms
    for at in listOfInputAtoms:

###########################Give me an atom, active atom Type, O/P: allCLOSE Atoms##########################################
        struct = at.parent.parent.parent.parent

        ##provide the type of active atoms you need in this function and it will generate a custom list of 
        #all those possible atoms##This funct: get_DonorAcceptor has both known and unknown atoms-

        customList = stp.get_DonorAcceptorList(struct, aaType,debug=debug)

        #Get all the close atoms with the given custom list           
        allCloseAtoms = get_allCloseAtomInfoForOneAtom(at, customList, debug=debug)
        
        #Append to the List of close atoms
        LOCA.append(allCloseAtoms)
        #[0] index of allCloseAtoms provides number of close atoms including itselff!So net close atom = dim-1
        dim.append(np.shape(allCloseAtoms)[0])
        

    return LOCA, dim



def get_connectedToDonor_BB(atomDonor, debug=0):

    '''
    objective: get the hydrogen connected to donor atom for the backbone atom. 
               Atom donor in this case is the backbone Nitrogen!
               atom3: is the hydrogen connected to the donor atom

    Input: -atomDonor: the donor atom of concern

    Output:  -atom3: is the hydrogen connected to the donor atom(N)

    '''

    res = atomDonor.parent
    atom3 = [res['H']]
   
    return atom3




def get_connectedToDonor_ARG(atomDonor, debug=0):

    '''
    objective: get the atom3(hydrogen connected to donor atom) for Arginine 
               Atom donor in this case can be: NE/NH1/NH2
               atom3: is the hydrogen connected to the donor atom: HE/HH11,HH12/HH21,HH22
                
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HE/HH11,HH12/HH21,HH22)
    '''


    res = atomDonor.parent
    
    
    if(atomDonor.name == 'NE'):
        atom3 = [res['HE']]

    if(atomDonor.name == 'NH1'):
        atom3_1 = res['HH11'] ##HH1
        atom3_2 = res['HH12'] ##HH1
        atom3 = [atom3_1, atom3_2]

    if(atomDonor.name == 'NH2'):
        atom3_1 = res['HH21'] ##HH2
        atom3_2 = res['HH22'] ##HH2
        atom3 = [atom3_1, atom3_2]

    return atom3



def get_connectedToDonor_ASH(atomDonor, debug=0):

    '''
    objective: get the atom3(hydrogen connected to donor atom) for ASH
               Atom donor in this case can be: OD2
               atom3: is the hydrogen connected to the donor atom: HD2
                 
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HD2)

    '''

    res = atomDonor.parent
    atom3 = [res['HD2']]
    return atom3




def get_connectedToDonor_GLH(atomDonor, debug=0):

    '''
    objective: get the atom3 (atom connected to donor atom) for GLH
               Atom donor in this case can be: OE2
               atom3: is the hydrogen connected to the donor atom: HE2
                
    Input: -atomDonor: the donor atom of concern           
    Output:  -atom3: is the hydrogen connected to the donor atom(HE2)
    '''

    res = atomDonor.parent
    atom3 = [res['HE2']]

    return atom3




def get_connectedToDonor_ASN(atomDonor, debug=0):

    '''
    objective: get the atom3( hydrogen connected to donor atom) for ASN
               Atom donor in this case can be: ND2
               atom3: is the hydrogen connected to the donor atom: HD21, HD22
    
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HD21, HD22)
    '''
    res = atomDonor.parent

    atom3_1 = res['HD21'] #HD21
    atom3_2 = res['HD22'] #HD22

    atom3 = [atom3_1, atom3_2]

    return atom3


     
def get_connectedToDonor_GLN(atomDonor, debug=0):

    '''
    objective: get the atom3(hydrogen atom connected to donor atom) for GLN
               Atom donor in this case can be: NE2
               atom3: is the hydrogen connected to the donor atom: HE21, HE22
                 
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HE21, HE22)
    '''
    res = atomDonor.parent

    atom3_1 = res['HE21'] #HE21
    atom3_2 = res['HE22'] #HE22

    atom3 = [atom3_1, atom3_2]

    return atom3

     

def get_connectedToDonor_TYR(atomDonor, debug=0):

    '''
    objective: get the atom3(hydrogen atom connected to donor atom)for TYR

               Atom donor in this case can be: OH
               atom3: is the hydrogen connected to the donor atom: HH
                
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HH)

    '''
    res = atomDonor.parent
    atom3 = [res['HH']] #HH

    return atom3



def get_connectedToDonor_TRP(atomDonor, debug=0):

    '''
    objective: get the atom3 (hydrogen atom connected to donor atom) for TRP
               Atom donor in this case can be: NE1
               atom3: is the hydrogen connected to the donor atom: HE1

    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HE1)
    '''
    res = atomDonor.parent
    atom3 = [res['HE1']] #HE1

    return atom3


def get_connectedToDonor_HID(atomDonor, debug=0):
    '''
    objective: get the atom3(hydrogen connected to donor atom) for HID
               Atom donor in this case can be: ND1
               atom3: is the hydrogen connected to the donor atom: HD1
                    
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HD1)

    '''
    res = atomDonor.parent
    atom3 = [res['HD1']] #HD1
    return atom3


def get_connectedToDonor_HIE(atomDonor, debug=0):

    '''
    objective: get the atom3(hydrogen atom connected to donor atom) for HIE
               Atom donor in this case can be: NE2
               atom3: is the hydrogen connected to the donor atom: HE2

    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HE2)

    '''
    res = atomDonor.parent
    atom3 = [res['HE2']] #HE2
    return atom3


def get_connectedToDonor_HIP(atomDonor, debug=0):
    '''
    objective: get the atom3(hydrogen connected to the donor atom) for HIP

               Atom donor in this case can be: ND1/ NE2
               atom3: is the hydrogen connected to the donor atom: HE1/HE2
               Taking advantage of HID and HIP functions set up
                
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HE1/HE2)
    ''' 
    if(atomDonor.name == 'ND1'):
        atom3 = get_connectedToDonor_HID(atomDonor, debug=debug)
    
    if(atomDonor.name == 'NE2'):
        atom3 = get_connectedToDonor_HIE(atomDonor, debug=debug)
         
    return atom3


############################################SP3###################################################################3
def get_connectedToDonor_LYS(atomDonor, debug=0):

    '''
    objective: get the atom3(hydrogen atom connected to donor atom)for LYS
               Atom donor in this case can be: NZ
               atom3: is the hydrogen connected to the donor atom: HZ1,HZ2,HZ3
                    
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HZ1, HZ2, HZ3)
    '''
    res = atomDonor.parent
    atom3_1 = res['HZ1'] #HZ1
    atom3_2 = res['HZ2'] #HZ1
    atom3_3 = res['HZ3'] #HZ1

    atom3 = [atom3_1, atom3_2, atom3_3]
 
    return atom3


def get_connectedToDonor_SER(atomDonor, debug=0):

    '''
    objective: get the atom3(hydrogen atom connected to donor)for SER
               Atom donor in this case can be: OG
               atom3: is the hydrogen connected to the donor atom: HG
                
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HG)
    '''

    res = atomDonor.parent
    atom3 = [res['HG']] #HG

    return atom3

     
def get_connectedToDonor_THR(atomDonor, debug=0):

    '''
    objective: get the atom3(hydrogen atom connected to donor atom) for THR
               Atom donor in this case can be: OG1
               atom3: is the hydrogen connected to the donor atom: HG1
    
    Input:   -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HG1)
    '''
    res = atomDonor.parent
    atom3 = [res['HG1']] #HG1

    return atom3



def get_r_theta(atom1, atom2, atom3, hydNotAtom=0, debug=0):

    '''
    objective: To get r and theta(bond angle) 
                
    Input:  atom1, atom2 and atom3 all are atom type
            -hydNotAtom: is a flag that allows hydrogen to be treated as coords(list:[x,y,z]) and not atom!
            NOTE: -r is distance between atom 1 and atom2
                  -theta: is bond angle between atom1-atom2-atom3

    Output: -r: radial distance,
            -theta: bond angle in degrees

    '''
    atom1_vec = atom1.get_vector()
    atom2_vec = atom2.get_vector()
    if(hydNotAtom ==1):
        atom3_vec = Vector(atom3)##Hydrogen may not be added as an atom yet!
    else:
        atom3_vec = atom3[0].get_vector()

    #get radial distance
    r = atom1-atom2
    #compute bond angle in degrees
    theta = np.rad2deg(calc_angle( atom1_vec, atom2_vec, atom3_vec) )

    return r, theta



def get_gamma(donorAt_vec, acceptorAt_vec, lp_vec, debug=0): 
    '''
    objective: to compute angle related to the lone pair: angle between acceptor-donor vec and lone pair-acceptor vec
    input: donorAt_vec: donor atom coord as a vector
           acceptorAt_vec: acceptor atom coord as a vector
           lp_vec: LP atom coord as vector
    output: gamma:angle between acceptor-donor vec and lone pair-acceptor vec in degrees
    '''
    acceptorDonor_vec = donorAt_vec - acceptorAt_vec
    acceptorLP_vec = lp_vec - acceptorAt_vec
    cosGamma = np.dot(acceptorLP_vec, acceptorDonor_vec)/(np.linalg.norm(acceptorLP_vec)*np.linalg.norm( acceptorDonor_vec)) 
    gamma = np.rad2deg(np.arccos(cosGamma))
    return gamma

def checkEnergyInRange(energy, stateSet = False, debug=0):
    '''
    objective: to keep energy value in check
    Input: -energy : value of the energy computed
           -stateSet: is a flag where one is setting a state with the given input energy value.
    Output:Exit if the energy value is above and state is being set. Else just give a warning!
    '''    
    if(abs(energy)>10):
        stp.append2log(f"Energy value found: {energy} is out of range. But the state is not yet set!\n", debug=0)
        if(stateSet ==True): 
            stp.append2log(f"Energy value: {energy} is out of range. Human intervention IS required. Exiting code \n", debug=0)
            os._exit(0)


def get_atomH_energy(atomDonor,  debug=0):
    '''
    objective: To get connected(/bonded) atoms of a donor. Specifically- the hydrogen atom, and the other connected atom 
    Input:-atomDonor: The donor atom
    Output: -atom3:the connected hydrogen
    ''' 
    ##check for the backbone nitrogen as a donor-else look into side chain atoms
    if(atomDonor.name == 'N'):
       atom3 = get_connectedToDonor_BB(atomDonor, debug=debug)
       return atom3
    else: 
        get_connectedAtmsForDonor = {'ARG': get_connectedToDonor_ARG,
                                     'ASN':get_connectedToDonor_ASN, 'ASNR':get_connectedToDonor_ASN,
                                    'GLN': get_connectedToDonor_GLN, 'GLNR': get_connectedToDonor_GLN,
                                    'TYR': get_connectedToDonor_TYR, 'TRP': get_connectedToDonor_TRP, 
                                    'HID': get_connectedToDonor_HID, 'HIDR': get_connectedToDonor_HID,
                                    'HIE': get_connectedToDonor_HIE, 'HIER': get_connectedToDonor_HIE,
                                    'HIP': get_connectedToDonor_HIP, 'HIPR': get_connectedToDonor_HIP,
                                    'LYS': get_connectedToDonor_LYS, 'SER': get_connectedToDonor_SER,
                                    'THR':get_connectedToDonor_THR, 'ASH':get_connectedToDonor_ASH,
                                    'GLH':get_connectedToDonor_GLH}


        try:
          atom3 =  get_connectedAtmsForDonor[atomDonor.parent.resname](atomDonor, debug=0)
        except KeyError:
          stp.append2log(f"No ATOM3 for: {atomDonor.parent.resname} and {atomDonor.parent.id[1]} reason: either insufficent atoms or not found in dict \n", debug=0 )
          atom3 =[]
    return atom3



def computeEnergyAsAcceptor(acceptorAt, lp_vec, donorAt, attractive = 1, atype = 'SP3',chV_levelVal ='level_00_chV_00_structureNum_00', debug=0 ):

    ''' 
    objective: To compute energy for an acceptor atom
    input:  -acceptorAt: current acceptor atom
            -lp_vec: lone pair vector associated with current atom
            -donorAt: donor atom associated with acceptor Atom
            -attractive: is it donor-acceptor(=1) or (acceptor-acceptor=0) or (donor-donor=0)
            -atype: is the acceptor atom:SP3/SP2
            -chV_levelVal: string defining changeValue+ level value-used for creating files

    output: - enValAcc: each energy value computed for the given acceptor(multiple lone pairs/ multiple hydrogens)
            -enSumAcc: sum of all energies associated with the acceptor atom
    '''

    struct = acceptorAt.parent.parent.parent.parent

    param = []

    if(debug==1):
        #write impt data to a file
        outputFolder = stp.get_outputFolderName(debug=debug)
        opFolder = f"./{outputFolder}/energyInfo_{mc.protID}/data_{chV_levelVal}"
        checkFolderPresent = os.path.isdir(opFolder)
        if not checkFolderPresent: os.makedirs(opFolder)
        accFname = f'{opFolder}/{acceptorAt.parent.id[1]}_{acceptorAt.parent.resname}_chain_{acceptorAt.parent.parent.id}.csv'
        fe = exists(accFname)
        if(fe ==False):
            param.append([ 'refAtom', 'refParent', 'atm1Parent', 'atm2Parent', 'atm1', 'atm1Coord', 'atm2', 'atm2Coord', 'atm3', 'atm3Coord','LP','LPCoord', 'r','theta','gamma', 'energy', 'attractive', 'atype', 'allCloseAtoms', 'CloseAtoms:unknownRes', 'refAtom:Rotamer'])#
     
    enValAcc = []
    enSumAcc = 0

    ##Only known atoms are picked up here
    customList = stp.get_knownDonorAcceptorListWRTOneAtom(struct, acceptorAt, aaType = 'DONOR_ACCEPTOR_BOTH', debug=debug)
    allCloseAtoms = get_allCloseAtomInfoForOneAtom(acceptorAt, customList, debug=debug)
    ##ONLY UNKNOWN LIST are picked up here!!ONLY to write2file!
    closeAtmsTBDList= stp.get_unknownDonorAcceptorListWRTOneAtom(struct, acceptorAt, aaType = 'DONOR_ACCEPTOR_BOTH_TBD',debug=debug)

    #close atoms that are not fixed yet
    if(closeAtmsTBDList):
        closeAtmsTBD = get_allCloseAtomInfoForOneAtom(acceptorAt, closeAtmsTBDList, debug = 0)
    else:
        closeAtmsTBD = "NONE"

    donorAt_vec = donorAt.get_vector() ##Atom2
    atom1 = acceptorAt
    atom2 = donorAt
    

    gamma = get_gamma(donorAt_vec, acceptorAt.get_vector(), lp_vec, debug=debug)

##############################COMPUTE ENERGY################################################################
    if(attractive==1):
        hPresent =0
        ##check if the backbone nitrogen has hydrogen present-as often it doesn't (due to being the first atom of the chain, etc)
        if(atom2.id == 'N'):
           hPresent =-1
           for at in atom2.parent.child_list:
                if(at.id == 'H'):
                    hPresent =1
                    break

        if(hPresent == -1): 
            stp.append2log(f"Donor atom:{atom2} belongs to res {atom2.parent} of chain: {atom2.parent.parent} has no hydrogen present \n", debug=0)

        else:
            atom3 = get_atomH_energy(atom2, debug=debug) 
            #for various hydrogen atoms sticking out of donor (theta will change)
            try: atom3
            except NameError:
                stp.append2log(f"Insufficent atoms/PSEUDO LPs or info not found in any funcs \n", debug=0)

            for at in range(np.shape(atom3)[0]):

                r, theta = get_r_theta(atom1, atom2, [atom3[at]], debug=debug)
                energy = hbond_energy(r, theta, gamma, attractive)
                checkEnergyInRange(energy, debug=debug)

                param.append([ acceptorAt, acceptorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, atom3[at], atom3[at].coord,lp_vec,lp_vec, r,theta, gamma, energy, attractive, atype, allCloseAtoms, closeAtmsTBD, acceptorAt.parent.isRotamer])

                enValAcc.append([energy])
                enSumAcc = enSumAcc+energy                    
    else:
        #Repulsion:for going over multiple Lone pairs(behaving as hydrogen) sticking out of close atom acceptor (theta will change). Fake hydrogen coordinates change-theta changes. Iterate over multiple LPs(acting as hydrogen for computation). These LPs stick out of close atom acceptor. Close atom acceptor is the fake donor!
        #For atom3: get LP names,iterate over for energy calc.Closest atom is a fake DONOR but actually is an acceptor 
    #    #print(f"I am an ACCEPTOR being repulsive: {acceptorAt}, fake donor: {donorAt}, attractive :{attractive}")
        ##"donorAT" will be an actual acceptor atom-pretending to be a donor! This pretend donor atom is called pseudo acceptor as it is an acceptor atom pretending to be donor. The "atom3"/Hydrogen attached to this will be corresponding to the lone pair attached to this pretend donor atom.

        myPsAcceptor = mra.myAtom(donorAt) 
        LPNames = copy.deepcopy(myPsAcceptor.get_LPNames())
        #Get various LPs of the acceptor-Gamma will change
        #check if LNames associated with pseudo acceptor are actually present in the child list of pseduo acceptor's parent 
   
        psAcceptorParentChildNames = []

        for at in donorAt.parent.child_list:
            psAcceptorParentChildNames.append(at.name)
        
        
        LPNamesIter = copy.deepcopy(LPNames)
        for lp in LPNamesIter:
            if(lp not in psAcceptorParentChildNames):
                LPNames.remove(lp)
    
        if(not LPNames):
            stp.append2log(f"##################################In computeEnergyAsAcceptor, NO LPs found for acceptor atom:{acceptorAt} of {acceptorAt.parent} and psAcceptor atom: {myPsAcceptor} of {myPsAcceptor.parent}, as attractive:{attractive} #################################", debug=0) 
            param.append([ acceptorAt, acceptorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, 'LPNames[k]', 'atom3Coord', 'lp_vec', 'lp_vec',  'r', 'theta', 'gamma', 'energy', attractive, atype, allCloseAtoms,closeAtmsTBD, acceptorAt.parent.isRotamer]) 
            
        else:
            for k in range(len(LPNames)):
                atom3Coord = donorAt.parent[LPNames[k]].coord
                #Atom3 is pretend donor
                r, theta = get_r_theta(atom1, atom2, atom3Coord,  hydNotAtom = 1, debug=debug)
                energy = hbond_energy(r, theta, gamma, attractive)
                checkEnergyInRange(energy, debug=debug)
                param.append([ acceptorAt, acceptorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, LPNames[k], atom3Coord, lp_vec, lp_vec,  r, theta, gamma, energy, attractive, atype, allCloseAtoms,closeAtmsTBD, acceptorAt.parent.isRotamer]) #removing phi

                enValAcc.append([energy])
                enSumAcc = enSumAcc+energy    


    if(debug==1):
        fileAcceptor = open(accFname,"a")    
        writerAcceptor = csv.writer(fileAcceptor)
        for row in param:
            writerAcceptor.writerow(row)
        fileAcceptor.close()


    return enValAcc, enSumAcc


def computeEnergyAsDonor(acceptorAt, hh_coord, donorAt, attractive = 1, atype = 'SP3', hName = 'H', chV_levelVal ='level_00_chV_00_structureNum_00', debug =0):
   
    ''' 
    objective: To compute energy for a donor atom
    input:  -acceptorAt: current acceptor atom
            -hh_Coord: associated hydrogen coord
            -donorAt: current donor atom:
            -attractive: is it donor-acceptor(=1) or (acceptor-acceptor=0) or (donor-donor=0)
            -atype: is the acceptor atom:SP3/SP2
            -hName: name of the hyd coord. If atom is not fixed use default hydrogen: 'H'
            -chV_levelVal: string defining changeValue+ level value-used for creating files

    output: - enValAcc: each energy value computed for the given acceptor(multiple lone pairs/ multiple hydrogens)
            -enSumAcc: sum of all energies associated with the acceptor atom
    '''
   
    
    struct = donorAt.parent.parent.parent.parent
    param = []
    if(debug==1):
        outputFolder = stp.get_outputFolderName(debug=debug)
        opFolder = f"./{outputFolder}/energyInfo_{mc.protID}/data_{chV_levelVal}"
        checkFolderPresent = os.path.isdir(opFolder)
        if not checkFolderPresent: os.makedirs(opFolder)
        donorFname = f'{opFolder}/{donorAt.parent.id[1]}_{donorAt.parent.resname}_chain_{donorAt.parent.parent.id}.csv'

        fe = exists(donorFname)
        if (fe ==False):
            param.append([ 'refAtom', 'refParent', 'atm1Parent', 'atm2Parent', 'atm1', 'atm1Coord', 'atm2', 'atm2Coord', 'atm3', 'atm3Coord','LP','LPCoord', 'r','theta', 'gamma', 'energy', 'attractive', 'atype','allCloseAtoms', 'CloseAtoms:unknownRes', 'refAtom:Rotamer'])
##
    #collect energy values for the donor atom            
    enValDon = []
    enSumDon = 0


    ##Only known atoms are picked up here
    customList = stp.get_knownDonorAcceptorListWRTOneAtom(struct,  donorAt, aaType = 'DONOR_ACCEPTOR_BOTH', debug=debug)
    allCloseAtoms = get_allCloseAtomInfoForOneAtom(donorAt, customList, debug=debug)

    ##ONLY UNKNOWN LIST to write2file!##WRTOneAtom allows the atoms of self residues to be considerd as known
    closeAtmsTBDList = stp.get_unknownDonorAcceptorListWRTOneAtom(struct, donorAt, aaType = 'DONOR_ACCEPTOR_BOTH_TBD', debug=debug)
    
    #close atoms that are not fixed yet
    if(closeAtmsTBDList):
        closeAtmsTBD = get_allCloseAtomInfoForOneAtom(donorAt, closeAtmsTBDList, debug=debug)
    else:
        closeAtmsTBD = "NONE"

    myAcceptorAt = mra.myAtom(acceptorAt)
    ##FOR attractive =0 : number of LPs = number of Hydrogen attached->atom3
    acceptorAt_vec = acceptorAt.get_vector()##Atom1
    atom1 = acceptorAt
    atom2 = donorAt
    atom3 = hh_coord
    #now only need gamma-can be multiple depending on surrounding lone pairs! 
    r, theta = get_r_theta(atom1, atom2, atom3, hydNotAtom=1, debug=debug)
    enSumDon = 0

########################### Compute Gamma and then energy!######################################################
    if(attractive ==1):
        LPNames = copy.deepcopy(myAcceptorAt.get_LPNames())
        #Get various LPs of the acceptor-Gamma will change. Create list of names of atoms in acceptorAt:
        acceptorParentChildNames = []

        #Get various LPs of the acceptor-Gamma will change. Check if LNames associated with acceptor are actually present in the child list of acceptor's parent 
        for at in acceptorAt.parent.child_list:
            acceptorParentChildNames.append(at.name)
 
        LPNamesIter = copy.deepcopy(LPNames)
        for lp in LPNamesIter:
            if(lp not in acceptorParentChildNames):
                LPNames.remove(lp)
    
        #If no LPNames are present-skip the computation.
        if(not LPNames):
            stp.append2log(f"##########################In computeEnergyAsDonor: NO LPs found for acceptor atom:{acceptorAt} of {acceptorAt.parent} and donor atom: {donorAt} of {donorAt.parent} as attractive:{attractive} ################################", debug=0)
            param.append([donorAt, donorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, hName, atom3,'LPNames[k]','LP_coord',r,theta, 'gamma', 'energy', attractive, atype, allCloseAtoms, closeAtmsTBD, donorAt.parent.isRotamer])
        else:
            ##you need to check before computing hbond_energy-figure out if the information is sufficient. if not-put 0.0!!
            for k in range(len(LPNames)):
                LP_coord = acceptorAt.parent[LPNames[k]].coord
                LP_vec = Vector(LP_coord)
                gamma = get_gamma(donorAt.get_vector(), acceptorAt_vec, LP_vec, debug=debug)
                energy = hbond_energy(r, theta, gamma, attractive=True)
                checkEnergyInRange(energy, debug=debug)
                enValDon.append([energy])
                enSumDon = enSumDon+energy 
                param.append([donorAt, donorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, hName, atom3,LPNames[k],LP_coord, r,theta, gamma, energy, attractive, atype, allCloseAtoms, closeAtmsTBD, donorAt.parent.isRotamer]) 
    else:
        #In Repulsion. Atom1 is actually a donor-getting associated-Hydrogen which I will treat as LPs
        #check if atom1.parent.child_list has 'H' in it:        
        hPresent =0
         ##check if the backbone nitrogen has hydrogen present-as often it doesn't (due to being the first atom of the chain, etc)
        if(atom1.id == 'N'):
            hPresent =-1
            for at in atom1.parent.child_list:
                if(at.id == 'H'):
                    hPresent =1
                    break

        if(hPresent ==-1):            
            stp.append2log(f"PS Donor(or acceptor) atom:{atom1} belongs res {atom1.parent} of its chain: {atom1.parent.parent} has NO hydrogen present", debug=0)
        else:
            ps_atom3 = get_atomH_energy(atom1, debug=debug)

            try: ps_atom3
            except NameError:
                stp.append2log(f"Insufficent atoms/Pseudo LPs or info not found in any funcs \n", debug=0)
            
            #iterate over multiple hydrogens (acting as LPs for computation)!
            for at in range(np.shape(ps_atom3)[0]):

                LP_coord = ps_atom3[at].coord 
                LP_vec = Vector(LP_coord)
                gamma = get_gamma(donorAt.get_vector(), acceptorAt_vec, LP_vec, debug=debug)
                energy = hbond_energy(r, theta, gamma, attractive)
                checkEnergyInRange(energy, debug=debug)
                
                
                param.append([donorAt, donorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, hName, atom3,ps_atom3[at],LP_coord,  r,theta, gamma, energy, attractive, atype, allCloseAtoms, closeAtmsTBD, donorAt.parent.isRotamer])
                           
                enValDon.append([energy])
                enSumDon = enSumDon+energy

    if(debug==1):
        fileDonor = open(donorFname,"a")    
        writerDonor = csv.writer(fileDonor)
        for row in param:
            #print(row)
            writerDonor.writerow(row)
        fileDonor.close()


    return enValDon, enSumDon


