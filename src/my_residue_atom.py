import Bio
import numpy as np
import code

from collections import namedtuple

import setup_protein as stp
import my_constants as mc

##This file contains the two class data structures used widely in the code based on residue and atom. Thus the two classes are: my_residue and my_atom. These are based off the biopython Residue(which is also a class) and biopython Atom(another class)
############################################################
#### MY RESIDUE CLASS!!
########################################################################################################################

class my_residue(Bio.PDB.Residue.Residue):
    '''
    Defining a class of my_residue that inherits biopython residue 

    '''

    def __init__(self, residue):
        Bio.PDB.Residue.Residue.__init__(self, residue.id, residue.resname, residue.segid)
        self.residue = residue
        self.idvalue = residue.id 
        self.isKnown = 32 #initiates as an integer value 32-therefore not 0 or 1. if 0: its unknown, and if 1-its known
        self.child_list = residue.get_list()

    def get_name(self):
        '''Get residue name '''
        return self.resname


##This function defines only the "valid" amino acid or which has the first id space empty.
#Looks like this can happen for metal atoms too. So PLS use validResnames
##CHECK IF THIS WORKS!!
    def is_valid_amino_acid(self):
        ''' Defining the valid Amino acids that not the Hetero-residues or water.'''
        return 1 if (self.idvalue[0] == ' ' and self.resname in mc.validResnames) else 0


###If you are at any of the residue of concen/unknown residue is used interchangeably in this code
    def is_residue_of_concern(self):
        ''' Defining if the residue is in the residue of concern list as defined in the myConstants class'''
        return 1 if self.resname in mc.res_of_concern else 0

##defining the active atoms of unknown residue
    def get_unknown_residue_acceptor_donor_atoms(self):
        '''
            objective: gets the active atom(donor/acceptor) of the unknown residues
            input: self residue
            output: -active atoms: list of donor/acceptor atoms
        '''

        if(self.resname == 'HIS' or self.resname == 'HIP' or self.resname == 'HIE' or self.resname == 'HID'
                                or self.resname == 'HIPR' or self.resname == 'HIER' or self.resname == 'HIDR'):
            activeAtoms = [self.residue['ND1'], self.residue['NE2']]

        elif(self.resname == 'GLN' or self.resname == 'GLNR'):
            activeAtoms = [self.residue['OE1'], self.residue['NE2']]

        elif(self.resname == 'ASN' or self.resname == 'ASNR'):
            activeAtoms = [self.residue['OD1'], self.residue['ND2']]

        elif(self.resname == 'ASH' or self.resname == 'ASP'):
            activeAtoms = [self.residue['OD1'], self.residue['OD2']]

        elif(self.resname == 'GLH' or self.resname == 'GLU'):
            activeAtoms = [self.residue['OE1'], self.residue['OE2']]
        else:
            activeAtoms = []

        return activeAtoms
 
############################################################
#### MY ATOM CLASS!!
########################################################################################################################

class my_atom(Bio.PDB.Atom.Atom): 

    '''
    Defining a class of my_atom that inherits biopython atom

    '''

    def __init__(self, atom):
        
        self.name = atom.name 
        self.id = atom.id
        self.full_id = atom.full_id
        self.parent = atom.get_parent()
        self.coord = atom.coord
        self.bfactor = atom.get_bfactor()
        self.coord_vector = atom.get_vector()
        self.hbondNumber = 0
        self.donorNumber = 0
        self.acceptorNumber = 0
        self.atomBehavior = self.set_default_behavior()
        

# Figure out if atom is part of backbone or not
    def is_backbone(self):
       ''' O/P: gives info regarding if provided atom is part of backbone.
            Note:I am counting Proline's N as part of backbone '''   

       return 1 if self.name in mc.atoms_backbone else 0
       

    def get_id(self):
        ''' O/P: provides the id of the atom which is same as the name of the atom'''
        return self.id

    def get_bfactor(self):
        ''' O/P: provides the bfactor value of the atom'''
        return self.bfactor

#Get the name of parent residue of the atom
    def get_atom_parent_name(self):
        ''' Get the name of parent residue of the atom'''
        return self.parent.resname


    def set_default_behavior(self):
            '''O/P: Provides the behavior of atom as 
               donor/acceptor/both/none/TBD based on the name of the atom and parent name  '''

            atomParentName = self.get_atom_parent_name()
            
            if(self.name[0] == 'N' and (atomParentName != 'PRO')):
                ##ANY Nitrogen is a donor-except Proline backbone and HIS. In HIS: it can be an acceptor!!
                self.atomBehavior = mc.donor

            elif (self.name[0] == 'O' and self.name!= 'OXT'):
                self.atomBehavior = mc.acceptor
            else:
                self.atomBehavior = mc.doesNothing
        
            if (self.name == 'ND1' and atomParentName == 'HIE') or (self.name == 'NE2' and atomParentName == 'HID'):
                self.atomBehavior = mc.acceptor

            if (atomParentName == 'HIS') and  (self.name == 'ND1' or self.name == 'NE2'):
                self.atomBehavior = mc.TBD

            if( ((atomParentName in mc.res_with_OH ) and (self.name == 'OG1' or self.name =='OG' or self.name == 'OH'))\
              or( atomParentName == 'GLH' and self.name == 'OE2') or (atomParentName == 'ASH' and self.name == 'OD2')):
                self.atomBehavior = mc.both

            return self.atomBehavior 

#Behavior of the atom: Donor or acceptor or nothing      
    def get_behavior(self):
        '''O/P: Provides the behavior of atom as 
           donor/acceptor/both/none based on the name of the atom and parent name  '''
        return self.atomBehavior ##This goes to default behavior


    def get_lonepair_names(self):
        '''This method provides the lone pair names(ex."LP1","LP2","LP3",etc) associated with the given atom'''

        if(self.is_backbone()==1):
            LPNames = mc.LPBB[0]
            return LPNames

        dict_LPSCName = {'OD1_ASP': mc.LPSCnamesASP[0],  'OD2_ASP': mc.LPSCnamesASP[1],
                          'OE1_GLU':mc.LPSCnamesGLU[0], 'OE2_GLU':mc.LPSCnamesGLU[1],
                          'OD1_ASN': mc.LPSCnamesASN[0],'OE1_GLN': mc.LPSCnamesGLN[0],
                          'OD1_ASNR': mc.LPSCnamesASN[0],'OE1_GLNR': mc.LPSCnamesGLN[0],
                          'ND1_HIE': mc.LPSCnamesHIE[0],'NE2_HID': mc.LPSCnamesHID[0],
                          'ND1_HIER': mc.LPSCnamesHIE[0],'NE2_HIDR': mc.LPSCnamesHID[0],
                          'OG_SER':mc.LPSCnamesSER[0], 'OG1_THR':mc.LPSCnamesTHR[0],
                          'OH_TYR':mc.LPSCnamesTYR[0], 'OD1_ASH':mc.LPSCnamesASH[0],
                          'OD2_ASH':mc.LPSCnamesASH[1], 'OE1_GLH':mc.LPSCnamesGLH[0],
                          'OE2_GLH':mc.LPSCnamesGLH[1]
        }
        try:
            LPNames = dict_LPSCName[self.name+'_'+self.parent.resname]
        
        except KeyError:
            stp.append_to_log(f"No LP for: {self.name} of {self.parent} and chain: {self.parent.parent}\n", debug=0)
            LPNames = None
        
        return LPNames


    def get_lonepairs_atoms(self):
        '''
        This method provides with the given atom's associated lone pairs
        '''
        
        namesLP = self.get_lonepair_names()
        LPAtoms = []

        for name in namesLP:
            try:
                LPAtoms.append(self.parent[name])
            except KeyError:
                stp.append_to_log(f"No LP: {name} for: {self.parent} with ID: {self.parent.id[1]} and chain: {self.parent.parent}\n", debug=0 )
                pass

        return LPAtoms


    
    def get_hydrogen_names(self):

        '''
        This method provides with the names of associated hydrogen atoms to a given atom
        '''

        if(self.is_backbone()==1):
            hydNames = mc.HBB[0]
            return hydNames

        dict_HSCName = {'NE_ARG': mc.HSCnamesARG[0], 'NH1_ARG': mc.HSCnamesARG[1], 'NH2_ARG': mc.HSCnamesARG[2],
                        'OD2_ASH': mc.HSCnamesASH[0], 'ND2_ASN': mc.HSCnamesASN[0],
                        'OE2_GLH': mc.HSCnamesGLH[0], 'NE2_GLN': mc.HSCnamesGLN[0],
                        'NE2_HIE': mc.HSCnamesHIE[0], 'ND1_HID': mc.HSCnamesHID[0], 
                        'ND1_HIP': mc.HSCnamesHIP[0], 'NE2_HIP': mc.HSCnamesHIP[1],
                        'NZ_LYS': mc.HSCnamesLYS[0], 'OG_SER': mc.HSCnamesSER[0],
                        'OG1_THR': mc.HSCnamesTHR[0], 'OH_TYR': mc.HSCnamesTYR[0],
                        'NE1_TRP': mc.HSCnamesTRP[0]
        }
        
        try:
            hydNames = dict_HSCName[self.name+'_'+self.parent.resname]
        
        except KeyError:
            stp.append_to_log(f"No H for: {self.name} of {self.parent} and chain {self.parent.parent}\n", debug=0 )
            hydNames = None
        
        return hydNames


