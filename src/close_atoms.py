"""
This module contains computations related to close atoms in a given residue.

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



import os, sys
import csv
import Bio
import numpy as np

import copy

from os.path import exists
from Bio.PDB import *



import setup_protein as stp
import my_residue_atom as mra
import my_constants as mc


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
            return mc.repulsive_k/(r**(mc.r_power))
    else:
        return 0.0


def get_close_atom_distance_info(atomCurr, close_atoms, include_self=True):

    """
        objective: To provide a list and information regarding close atoms,
                   close atom parent, close atom behavior and distance
                   from current atom
        I/P:
             atomCurr: Current Atom,
             close_atoms: the list of close atoms you want information about
             include_self: whether to include the atoms in the same residue as atomCurr
        O/P: Distance information in ascending order
    """

    distInfo = []
    if not include_self:
        target_resid = atomCurr.get_parent().id[1]

    for atomClose in close_atoms:
       aClose = mra.my_atom(atomClose)
       dist = atomClose-atomCurr
       resid = atomClose.get_parent().id[1]
       if not include_self and resid == target_resid:
           continue
       distInfo.append([atomClose, atomClose.get_parent().id[1],atomClose.get_parent().resname, aClose.get_behavior().abbrev, dist])

    
    #sorting the array according to distance: smallest to largest
    distInfo.sort(key = lambda i: i[-1])
   
    return distInfo

def get_all_close_atom_info_for_one_atom(target_atom, givenList, dist_cutoff=mc.deltaD, include_self=True):

    """
        I/P: 
            -aCurr: current atom
            -searchListName: search list 
            -givenList
            include_self: whether to include the atoms in the same residue as target_atom
        O/P: A close atom matrix has first row as an active atom and its related info in the columns
        All atoms within the DeltaD radius is listed in the next few rows, with the closest
        atom being on row 1. The atoms are listed in closest to farthest order.        
    """
    
    if(not givenList): 
        sys.exit("No search list provided for getting close atoms")

    ns = Bio.PDB.NeighborSearch(givenList)
    close_atoms = ns.search(target_atom.coord, dist_cutoff) ##distance cut off is defined by mc.deltaD
    
    if include_self:
        if(target_atom not in close_atoms):
            close_atoms.append(target_atom)

    if np.shape(close_atoms)[0] > 0:
        distInfo = get_close_atom_distance_info(target_atom, close_atoms, include_self=include_self)
    else:
        return []

    return distInfo


def get_list_of_close_atoms_for_list_of_atoms(listOfInputAtoms,  aaType, include_self=True):
    
    """
    Objective: To get close atoms for a list of atoms.

    Input: -listOfInputAtoms: list of input atoms you want close atoms for
           -aatype: type of active atoms for list of close atoms, includes:
                    -DONOR: only donor atoms
                    -ACCEPTOR: only acceptor atoms
                    -DONOR_ACCEPTOR_BOTH: only donor/
                    -DONOR_ACCEPTOR_BOTH_TBD: Donor,acceptor, both and to be determined
                    #NOTE: the atoms may not necessarily be correctly placed-for example
                           ASN may not be set, yet it has donors and acceptors.
            include_self: whether to include the atoms in the same residue as target_atom
    Output:-LOCA: List of Close Atoms
           -dim: Number of Close Atoms including its own atom and all the info provided with it..i.e:
            [atom, atom.parent.id, atom.parent.resname, distance from original atom]

    """

    #create list of close atoms and dimensions
    LOCA = []
    dim = []

    #iterate over all atoms in the list of input atoms
    for at in listOfInputAtoms:
        struct = at.parent.parent.parent.parent
        customList = stp.get_donor_acceptor_list(struct, aaType)

        #Get all the close atoms with the given custom list           
        allCloseAtoms = get_all_close_atom_info_for_one_atom(at, customList, include_self=include_self)
        
        LOCA.append(allCloseAtoms)
        dim.append(np.shape(allCloseAtoms)[0])
        

    return LOCA, dim


def get_hydrogen_connected_to_donor_backbone(atomDonor):


    """
    objective: get the hydrogen connected to donor atom for the backbone atom. 
               Atom donor in this case is the backbone Nitrogen
               atom3: is the hydrogen connected to the donor atom

    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(N)
    """


    res = atomDonor.parent
    atom3 = [res['H']]
   
    return atom3




def get_hydrogen_connected_to_donor_ARG(atomDonor):

    """
    objective: get the atom3(hydrogen connected to donor atom) for Arginine 
               Atom donor in this case can be: NE/NH1/NH2
               atom3: is the hydrogen connected to the donor atom: HE/HH11,HH12/HH21,HH22
                
    Input:   -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HE/HH11,HH12/HH21,HH22)
    """


    res = atomDonor.parent
    
    
    if(atomDonor.name == 'NE'):
        atom3 = [res['HE']]

    if(atomDonor.name == 'NH1'):
        atom3_1 = res['HH11']
        atom3_2 = res['HH12']
        atom3 = [atom3_1, atom3_2]

    if(atomDonor.name == 'NH2'):
        atom3_1 = res['HH21']
        atom3_2 = res['HH22']
        atom3 = [atom3_1, atom3_2]

    return atom3



def get_hydrogen_connected_to_donor_ASH(atomDonor):

    """
    objective: get the atom3(hydrogen connected to donor atom) for ASH
               Atom donor in this case can be: OD2
               atom3: is the hydrogen connected to the donor atom: HD2
                 
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HD2)

    """

    res = atomDonor.parent
    atom3 = [res['HD2']]
    return atom3




def get_hydrogen_connected_to_donor_GLH(atomDonor):

    """
    objective: get the atom3 (atom connected to donor atom) for GLH
               Atom donor in this case can be: OE2
               atom3: is the hydrogen connected to the donor atom: HE2
                
    Input: -atomDonor: the donor atom of concern           
    Output: -atom3: is the hydrogen connected to the donor atom(HE2)
    """

    res = atomDonor.parent
    atom3 = [res['HE2']]

    return atom3




def get_hydrogen_connected_to_donor_ASN(atomDonor):

    """
    objective: get the atom3( hydrogen connected to donor atom) for ASN
               Atom donor in this case can be: ND2
               atom3: is the hydrogen connected to the donor atom: HD21, HD22
    
    Input: -atomDonor: the donor atom of concern
    Output:-atom3: is the hydrogen connected to the donor atom(HD21, HD22)
    """
    res = atomDonor.parent

    atom3_1 = res['HD21']
    atom3_2 = res['HD22']

    atom3 = [atom3_1, atom3_2]

    return atom3


     
def get_hydrogen_connected_to_donor_GLN(atomDonor):

    """
    objective: get the atom3(hydrogen atom connected to donor atom) for GLN
               Atom donor in this case can be: NE2
               atom3: is the hydrogen connected to the donor atom: HE21, HE22
                 
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HE21, HE22)
    """

    res = atomDonor.parent

    atom3_1 = res['HE21']
    atom3_2 = res['HE22']

    atom3 = [atom3_1, atom3_2]

    return atom3

     

def get_hydrogen_connected_to_donor_TYR(atomDonor):

    """
    objective: get the atom3(hydrogen atom connected to donor atom)for TYR
               Atom donor in this case can be: OH
               atom3: is the hydrogen connected to the donor atom: HH
                
    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HH)
    """
    res = atomDonor.parent
    atom3 = [res['HH']]

    return atom3



def get_hydrogen_connected_to_donor_TRP(atomDonor):

    """
    objective: get the atom3 (hydrogen atom connected to donor atom) for TRP
               Atom donor in this case can be: NE1
               atom3: is the hydrogen connected to the donor atom: HE1

    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HE1)
    """

    res = atomDonor.parent
    atom3 = [res['HE1']]

    return atom3


def get_hydrogen_connected_to_donor_HID(atomDonor):

    """
    objective: get the atom3(hydrogen connected to donor atom) for HID
               Atom donor in this case can be: ND1
               atom3: is the hydrogen connected to the donor atom: HD1
                    
    Input:   -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HD1)
    """

    res = atomDonor.parent
    atom3 = [res['HD1']]
    return atom3


def get_hydrogen_connected_to_donor_HIE(atomDonor):
    """
    objective: get the atom3(hydrogen atom connected to donor atom) for HIE
               Atom donor in this case can be: NE2
               atom3: is the hydrogen connected to the donor atom: HE2

    Input: -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HE2)
    """
    res = atomDonor.parent
    atom3 = [res['HE2']]
    return atom3


def get_hydrogen_connected_to_donor_HIP(atomDonor):

    """
    objective: get the atom3(hydrogen connected to the donor atom) for HIP

               Atom donor in this case can be: ND1/ NE2
               atom3: is the hydrogen connected to the donor atom: HE1/HE2
               Taking advantage of HID and HIP functions set up
                
    Input:   -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HE1/HE2)
    """

    if(atomDonor.name == 'ND1'):
        atom3 = get_hydrogen_connected_to_donor_HID(atomDonor)
    
    if(atomDonor.name == 'NE2'):
        atom3 = get_hydrogen_connected_to_donor_HIE(atomDonor)
         
    return atom3


############################################SP3###################################################################3
def get_hydrogen_connected_to_donor_LYS(atomDonor):

    """
    objective: get the atom3(hydrogen atom connected to donor atom)for LYS
               Atom donor in this case is: NZ
               atom3: is the hydrogen connected to the donor atom: HZ1,HZ2,HZ3
                    
    Input:   -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HZ1, HZ2, HZ3)
    """

    res = atomDonor.parent
    atom3_1 = res['HZ1'] 
    atom3_2 = res['HZ2'] 
    atom3_3 = res['HZ3'] 

    atom3 = [atom3_1, atom3_2, atom3_3]
 
    return atom3


def get_hydrogen_connected_to_donor_SER(atomDonor):

    """
    objective: get the atom3(hydrogen atom connected to donor)for SER
               Atom donor in this case is: OG
               atom3: is the hydrogen connected to the donor atom: HG
                
    Input: -atomDonor: the donor atom of concern
    Output: -atom3: is the hydrogen connected to the donor atom(HG)
    """

    res = atomDonor.parent
    atom3 = [res['HG']]

    return atom3

     
def get_hydrogen_connected_to_donor_THR(atomDonor):

    """
    objective: get the atom3(hydrogen atom connected to donor atom) for THR
               Atom donor in this case is: OG1
               atom3: is the hydrogen connected to the donor atom: HG1
    
    Input:   -atomDonor: the donor atom of concern
    Output:  -atom3: is the hydrogen connected to the donor atom(HG1)
    """

    res = atomDonor.parent
    atom3 = [res['HG1']]

    return atom3



def get_r_theta(atom1, atom2, atom3, hydNotAtom=0):

    """
    objective: To get r and theta(bond angle) 
                
    Input:  atom1, atom2 and atom3 all are atom type
            -hydNotAtom: is a flag that allows hydrogen to be treated as coords(list:[x,y,z]) and not atom
            NOTE: -r is distance between atom 1 and atom2
                  -theta: is bond angle between atom1-atom2-atom3

    Output: -r: radial distance,
            -theta: bond angle in degrees

    """
    atom1_vec = atom1.get_vector()
    atom2_vec = atom2.get_vector()
    if(hydNotAtom ==1):
        atom3_vec = Vector(atom3)
    else:
        atom3_vec = atom3[0].get_vector()

    #get radial distance
    r = atom1-atom2
    #compute bond angle in degrees
    theta = np.rad2deg(calc_angle( atom1_vec, atom2_vec, atom3_vec) )

    return r, theta



def get_gamma(donorAt_vec, acceptorAt_vec, lp_vec):
    
    """
    objective: to compute angle related to the lone pair: angle between
               acceptor-donor vector and lone pair-acceptor vector
    input: donorAt_vec: donor atom coord as a vector
           acceptorAt_vec: acceptor atom coord as a vector
           lp_vec: LP atom coord as vector
    output: gamma:angle between acceptor-donor vector, lone pair and
            acceptor vector in degrees
    """
    
    acceptorDonor_vec = donorAt_vec - acceptorAt_vec
    acceptorLP_vec = lp_vec - acceptorAt_vec
    cosGamma = np.dot(acceptorLP_vec, acceptorDonor_vec)/(np.linalg.norm(acceptorLP_vec)*np.linalg.norm( acceptorDonor_vec)) 
    gamma = np.rad2deg(np.arccos(cosGamma))
    return gamma


def check_energy_range(energy, stateSet = False, log_file=0):

    """
    objective: to keep energy value in check
    Input: -energy : value of the energy computed
           -stateSet: is a flag where one is setting a state with the 
                      given input energy value.
    Output:Exit if the energy value is above and state is being set.
           Else just give a warning in log file.
    """    

    if(abs(energy)>10):
        if(log_file):stp.append_to_log(f"Energy value found: {energy} is out of range. But the state is not yet set!\n")
        if(stateSet ==True ):
            if(log_file):stp.append_to_log(f"Energy value: {energy} is out of range. Human intervention IS required. Exiting code \n")
            os._exit(0)


def get_hydrogen_connected_to_donor(atomDonor, log_file=0):

    """
    objective: To get connected(/bonded) atoms of a donor. 
               Specifically- the hydrogen atom, and the other connected atom 
    Input:-atomDonor: The donor atom
    Output: -atom3:the connected hydrogen
    """
    ##check for the backbone nitrogen as a donor-else look into side chain atoms
    if(atomDonor.name == 'N'):
       atom3 = get_hydrogen_connected_to_donor_backbone(atomDonor)
       return atom3
    else: 
        get_connectedAtmsForDonor = {'ARG': get_hydrogen_connected_to_donor_ARG,
                                     'ASN':get_hydrogen_connected_to_donor_ASN, 'ASNR':get_hydrogen_connected_to_donor_ASN,
                                    'GLN': get_hydrogen_connected_to_donor_GLN, 'GLNR': get_hydrogen_connected_to_donor_GLN,
                                    'TYR': get_hydrogen_connected_to_donor_TYR, 'TRP': get_hydrogen_connected_to_donor_TRP, 
                                    'HID': get_hydrogen_connected_to_donor_HID, 'HIDR': get_hydrogen_connected_to_donor_HID,
                                    'HIE': get_hydrogen_connected_to_donor_HIE, 'HIER': get_hydrogen_connected_to_donor_HIE,
                                    'HIP': get_hydrogen_connected_to_donor_HIP, 'HIPR': get_hydrogen_connected_to_donor_HIP,
                                    'LYS': get_hydrogen_connected_to_donor_LYS, 'SER': get_hydrogen_connected_to_donor_SER,
                                    'LYN': get_hydrogen_connected_to_donor_LYS,
                                    'THR':get_hydrogen_connected_to_donor_THR, 'ASH':get_hydrogen_connected_to_donor_ASH,
                                    'GLH':get_hydrogen_connected_to_donor_GLH}


        try:
          atom3 =  get_connectedAtmsForDonor[atomDonor.parent.resname](atomDonor)
        except KeyError:
          if(log_file):stp.append_to_log(f"No ATOM3 for: {atomDonor.parent.resname} and {atomDonor.parent.id[1]} reason: either insufficent atoms or not found in dict \n")
          atom3 =[]
    return atom3


def get_info_for_acceptorAt_donorAt(inputAt):

    """
        objective: To get list of all close atoms(that are known) and close
                   atoms(including unknowns) for a given acceptor or donor atom.
        input:
            -inputAt: the acceptor or donor atom for which we want lists of known or unknown atoms
        
        output:
            -allCloseAtoms: all close atoms that have known behaviors(donor/acceptor/or can act as both)
            -closeAtmsTBD: all close atoms that have known and unknown behaviors
    """
    struct = inputAt.parent.parent.parent.parent
    ##Only known atoms are picked up here
    customList = stp.get_known_donor_acceptor_list_for_one_atom(struct, inputAt, aaType = 'DONOR_ACCEPTOR_BOTH')
    allCloseAtoms = get_all_close_atom_info_for_one_atom(inputAt, customList)
    ##Only unknown list are picked up here
    closeAtmsTBDList= stp.get_unknown_donor_acceptor_list_for_one_atom(struct, inputAt, aaType = 'DONOR_ACCEPTOR_BOTH_TBD')

    #close atoms that are not fixed yet
    if(closeAtmsTBDList):
        closeAtmsTBD = get_all_close_atom_info_for_one_atom(inputAt, closeAtmsTBDList)
    else:
        closeAtmsTBD = "NONE"

    return [allCloseAtoms, closeAtmsTBD]


def compute_energy_as_acceptor(acceptorAt, lp_vec, donorAt, attractive = 1, atype = 'SP3', log_file=0, debug=0,
                               pre_cal_acceptor_info=None):
 
    """
    objective: To compute energy for an acceptor atom
    input:  -acceptorAt: current acceptor atom
            -lp_vec: lone pair vector associated with current atom
            -donorAt: donor atom associated with acceptor Atom
            -attractive: is it donor-acceptor(=1) or (acceptor-acceptor=0) or (donor-donor=0)
            -atype: is the acceptor atom:SP3/SP2

    output: - enValAcc: each energy value computed for the given acceptor(multiple lone pairs/ multiple hydrogens)
            -enSumAcc: sum of all energies associated with the acceptor atom
    """
    


    param = []

    if(debug):
        outputFolder = stp.get_output_folder_name()
        #opFolder = f"./{outputFolder}/energyInfo_{mc.protID}/data_{chV_levelVal}"
        opFolder = f"./{outputFolder}/debug/energyInfo_{mc.protID}/data_{chV_levelVal}"
        checkFolderPresent = os.path.isdir(opFolder)
        if not checkFolderPresent: os.makedirs(opFolder)
        accFname = f'{opFolder}/{acceptorAt.parent.id[1]}_{acceptorAt.parent.resname}_chain_{acceptorAt.parent.parent.id}.csv'
        fe = exists(accFname)
        if(fe ==False):
            param.append([ 'refAtom', 'refParent', 'atm1Parent', 'atm2Parent', 'atm1', 'atm1Coord', 'atm2', 'atm2Coord', 'atm3', 'atm3Coord','LP','LPCoord', 'r','theta','gamma', 'energy', 'attractive', 'atype', 'allCloseAtoms', 'CloseAtoms:unknownRes', 'refAtom:Rotamer'])#
     
    enValAcc = []
    enSumAcc = 0

    if pre_cal_acceptor_info is None:
        [allCloseAtoms, closeAtmsTBD] = get_info_for_acceptorAt_donorAt(acceptorAt)
    else: 
        [allCloseAtoms, closeAtmsTBD] = pre_cal_acceptor_info

    donorAt_vec = donorAt.get_vector() ##Atom2
    atom1 = acceptorAt
    atom2 = donorAt
    

    gamma = get_gamma(donorAt_vec, acceptorAt.get_vector(), lp_vec)

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
            if(log_file): stp.append_to_log(f"Donor atom:{atom2} belongs to res {atom2.parent} of chain: {atom2.parent.parent} has no hydrogen present \n")

        else:
            atom3 = get_hydrogen_connected_to_donor(atom2)
            #for various hydrogen atoms bonded to donor (theta will change)
            try: atom3
            except NameError:
                if(log_file):stp.append_to_log(f"Insufficient atoms/pseudo LPs or info not found in any funcs \n")

            for at in range(np.shape(atom3)[0]):

                r, theta = get_r_theta(atom1, atom2, [atom3[at]])
                energy = hbond_energy(r, theta, gamma, attractive)
                check_energy_range(energy, log_file=log_file)

                param.append([ acceptorAt, acceptorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, atom3[at], atom3[at].coord,lp_vec,lp_vec, r,theta, gamma, energy, attractive, atype, allCloseAtoms, closeAtmsTBD, acceptorAt.parent.isRotamer])

                enValAcc.append([energy])
                enSumAcc = enSumAcc+energy                    
    else:
        
        myPsAcceptor = mra.my_atom(donorAt) 
        LPNames = copy.deepcopy(myPsAcceptor.get_lonepair_names())
        #Get various LPs of the acceptor-Gamma will change
        #check if LPNames associated with pseudo acceptor are actually present in the child list of pseduo acceptor's parent
   
        psAcceptorParentChildNames = []

        for at in donorAt.parent.child_list:
            psAcceptorParentChildNames.append(at.name)
        
        
        LPNamesIter = copy.deepcopy(LPNames)
        for lp in LPNamesIter:
            if(lp not in psAcceptorParentChildNames):
                LPNames.remove(lp)
    
        if(not LPNames):
            if(log_file): stp.append_to_log(f"##################################In compute_energy_as_acceptor, No Lone Pairs found for acceptor atom:{acceptorAt} of {acceptorAt.parent} and psAcceptor atom: {myPsAcceptor} of {myPsAcceptor.parent}, as attractive:{attractive} #################################")
            param.append([ acceptorAt, acceptorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, 'LPNames[k]', 'atom3Coord', 'lp_vec', 'lp_vec',  'r', 'theta', 'gamma', 'energy', attractive, atype, allCloseAtoms,closeAtmsTBD, acceptorAt.parent.isRotamer]) 
            
        else:
            for k in range(len(LPNames)):
                atom3Coord = donorAt.parent[LPNames[k]].coord
                #Atom3 is "donor"
                r, theta = get_r_theta(atom1, atom2, atom3Coord,  hydNotAtom = 1)
                energy = hbond_energy(r, theta, gamma, attractive)
                check_energy_range(energy, log_file=log_file)
                param.append([ acceptorAt, acceptorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, LPNames[k], atom3Coord, lp_vec, lp_vec,  r, theta, gamma, energy, attractive, atype, allCloseAtoms,closeAtmsTBD, acceptorAt.parent.isRotamer]) #removing phi

                enValAcc.append([energy])
                enSumAcc = enSumAcc+energy    


    if(debug):
        with open(accFname,"a") as fileAcceptor:
            writerAcceptor = csv.writer(fileAcceptor)
            for row in param:
                writerAcceptor.writerow(row)


    return enValAcc, enSumAcc


def compute_energy_as_donor(acceptorAt, hh_coord, donorAt, attractive = 1, atype = 'SP3', hName = 'H', log_file=0, debug =0,  pre_cal_donor_info=None):
   
    """
    objective: To compute energy for a donor atom
    input:  -acceptorAt: current acceptor atom
            -hh_Coord: associated hydrogen coordinate
            -donorAt: current donor atom:
            -attractive: is it donor-acceptor(=1) or (acceptor-acceptor=0) or (donor-donor=0)
            -atype: is the acceptor atom:SP3/SP2
            -hName: name of the hydrogen coord. If atom is not fixed use default hydrogen: 'H'
            -pre_cal_donor_info: pre-compute information regarding the donor atom

    output: - enValAcc: each energy value computed for the given acceptor(multiple lone pairs/ multiple hydrogen)
            -enSumAcc: sum of all energies associated with the acceptor atom
    """
   

    param = []
    if(debug):
        outputFolder = stp.get_output_folder_name()
        #opFolder = f"./{outputFolder}/energyInfo_{mc.protID}/data_{chV_levelVal}"
        opFolder = f"./{outputFolder}/debug/energyInfo_{mc.protID}/data_{chV_levelVal}"
        checkFolderPresent = os.path.isdir(opFolder)
        if not checkFolderPresent: os.makedirs(opFolder)
        donorFname = f'{opFolder}/{donorAt.parent.id[1]}_{donorAt.parent.resname}_chain_{donorAt.parent.parent.id}.csv'

        fe = exists(donorFname)
        if (fe ==False):
            param.append([ 'refAtom', 'refParent', 'atm1Parent', 'atm2Parent', 'atm1', 'atm1Coord', 'atm2', 'atm2Coord', 'atm3', 'atm3Coord','LP','LPCoord', 'r','theta', 'gamma', 'energy', 'attractive', 'atype','allCloseAtoms', 'CloseAtoms:unknownRes', 'refAtom:Rotamer'])
##
    enValDon = []


    if pre_cal_donor_info is None:
        [allCloseAtoms, closeAtmsTBD] = get_info_for_acceptorAt_donorAt(donorAt)

    else: 
        [allCloseAtoms, closeAtmsTBD] = pre_cal_donor_info

    myAcceptorAt = mra.my_atom(acceptorAt)

    acceptorAt_vec = acceptorAt.get_vector()##Atom1
    atom1 = acceptorAt
    atom2 = donorAt
    atom3 = hh_coord
    #now only need gamma-can be multiple depending on surrounding lone pairs! 
    r, theta = get_r_theta(atom1, atom2, atom3, hydNotAtom=1)
    enSumDon = 0

    ## Compute Gamma and then energy###
    if(attractive ==1):
        LPNames = copy.deepcopy(myAcceptorAt.get_lonepair_names())
        #Get various LPs of the acceptor-Gamma will change. Create list of names of atoms in acceptorAt:
        acceptorParentChildNames = []

        #Get various LPs of the acceptor-Gamma will change. Check if LPNames associated with acceptor are actually present in the child list of acceptor's parent
        for at in acceptorAt.parent.child_list:
            acceptorParentChildNames.append(at.name)
 
        LPNamesIter = copy.deepcopy(LPNames)
        for lp in LPNamesIter:
            if(lp not in acceptorParentChildNames):
                LPNames.remove(lp)
    
        if(not LPNames):
            if(log_file): stp.append_to_log(f"##########################In compute_energy_as_donor: No Lone Pairs found for acceptor atom:{acceptorAt} of {acceptorAt.parent} and donor atom: {donorAt} of {donorAt.parent} as attractive:{attractive} ################################")
            param.append([donorAt, donorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, hName, atom3,'LPNames[k]','LP_coord',r,theta, 'gamma', 'energy', attractive, atype, allCloseAtoms, closeAtmsTBD, donorAt.parent.isRotamer])
        else:
            for k in range(len(LPNames)):
                LP_coord = acceptorAt.parent[LPNames[k]].coord
                LP_vec = Vector(LP_coord)
                gamma = get_gamma(donorAt.get_vector(), acceptorAt_vec, LP_vec)
                energy = hbond_energy(r, theta, gamma, attractive=True)
                check_energy_range(energy, log_file=log_file)
                enValDon.append([energy])
                enSumDon = enSumDon+energy 
                param.append([donorAt, donorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, hName, atom3,LPNames[k],LP_coord, r,theta, gamma, energy, attractive, atype, allCloseAtoms, closeAtmsTBD, donorAt.parent.isRotamer]) 
    else:
        hPresent =0

        if(atom1.id == 'N'):
            hPresent =-1
            for at in atom1.parent.child_list:
                if(at.id == 'H'):
                    hPresent =1
                    break

        if(hPresent ==-1):            
            if(log_file):stp.append_to_log(f"Pseudo Donor(or acceptor) atom:{atom1} belongs residue {atom1.parent} of its chain: {atom1.parent.parent} has NO hydrogen present")
        else:
            ps_atom3 = get_hydrogen_connected_to_donor(atom1)

            try: ps_atom3
            except NameError:
                if(log_file):stp.append_to_log(f"Insufficient atoms/Pseudo LPs or info not found in any funcs \n")
            
            #iterate over multiple hydrogen (acting as LPs for computation)
            for at in range(np.shape(ps_atom3)[0]):

                LP_coord = ps_atom3[at].coord 
                LP_vec = Vector(LP_coord)
                gamma = get_gamma(donorAt.get_vector(), acceptorAt_vec, LP_vec)
                energy = hbond_energy(r, theta, gamma, bool(attractive))
                check_energy_range(energy, log_file=log_file)
                
                
                param.append([donorAt, donorAt.parent, atom1.parent, atom2.parent, atom1, atom1.coord, atom2, atom2.coord, hName, atom3,ps_atom3[at],LP_coord,  r,theta, gamma, energy, attractive, atype, allCloseAtoms, closeAtmsTBD, donorAt.parent.isRotamer])
                           
                enValDon.append([energy])
                enSumDon = enSumDon+energy

    if(debug):
        with open(donorFname,"a") as fileDonor:
            writerDonor = csv.writer(fileDonor)
            for row in param:
                writerDonor.writerow(row)


    return enValDon, enSumDon


