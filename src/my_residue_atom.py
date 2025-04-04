"""
This contains the two classes(my_residue and my_atom) and its related methods.


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



import Bio
import setup_protein as stp
import my_constants as mc


############################################################
#### my_residue class
########################################################################################################################

class my_residue(Bio.PDB.Residue.Residue):
    
    """Defining a class of my_residue that inherits biopython residue """

    def __init__(self, residue):
        Bio.PDB.Residue.Residue.__init__(self, residue.id, residue.resname, residue.segid)
        self.residue = residue
        self.idvalue = residue.id 
        self.isKnown = 32 #initiates as an integer value 32-therefore not 0 or 1. if 0: its unknown, and if 1-its known
        self.child_list = residue.get_list()

    def get_name(self):
        """Get residue name """
        return self.resname


##This function defines only the "valid" amino acid or which has the first id space empty.
    def is_valid_amino_acid(self):
        """ Defining the valid Amino acids that not the Hetero-residues or water."""
        return 1 if (self.idvalue[0] == ' ' and self.resname in mc.validResnames) else 0



    def is_residue_of_concern(self):
        """ Defining if the residue is in the residue of concern list as defined in the myConstants class"""
        return 1 if self.resname in mc.res_of_concern else 0

##defining the active atoms of unknown residue
    def get_unknown_residue_acceptor_donor_atoms(self):
        """
            objective: gets the active atom(donor/acceptor) of the unknown residues
            input: self residue
            output: -active atoms: list of donor/acceptor atoms
        """

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
#### my_atom class
########################################################################################################################

class my_atom(Bio.PDB.Atom.Atom): 

    """ Defining a class of my_atom that inherits biopython atom"""


    def __init__(self, atom):
        
        self.name = atom.name 
        self.id = atom.id
        self.full_id = atom.full_id
        self.parent = atom.get_parent()
        self.coord = atom.coord
        self.bfactor = atom.get_bfactor()
        self.coord_vector = atom.get_vector()
        self.atomBehavior = self.set_default_behavior()
        

# Figure out if atom is part of backbone or not
    def is_backbone(self):

       """  gives info regarding if provided atom is part of backbone.
            Note:Treating Proline's N as part of backbone """

       return 1 if self.name in mc.atoms_backbone else 0
       

    def get_id(self):
        """ Provides the id of the atom which is same as the name of the atom"""
        return self.id

    def get_bfactor(self):
        """Provides the b-factor value of the atom"""
        return self.bfactor

    def get_atom_parent_name(self):
        """ Get the name of parent residue of the atom"""
        return self.parent.resname


    def set_default_behavior(self):
            """O/P: Provides the behavior of atom as 
               donor/acceptor/both/none/TBD based on the name of the atom and parent name  """

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
        """ Provides the behavior of atom as 
           donor/acceptor/both/none based on the name of the atom and parent name  """
        return self.atomBehavior ##This goes to default behavior


    def get_lonepair_names(self):
        """This method provides the lone pair names(ex."LP1","LP2","LP3",etc) associated with the given atom"""

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
            stp.append_to_log(f"No LP for: {self.name} of {self.parent} and chain: {self.parent.parent}\n")
            LPNames = None
        
        return LPNames


    def get_lonepairs_atoms(self):
        """This method provides with the given atom's associated lone pairs"""
        
        namesLP = self.get_lonepair_names()
        LPAtoms = []

        for name in namesLP:
            try:
                LPAtoms.append(self.parent[name])
            except KeyError:
                stp.append_to_log(f"No LP: {name} for: {self.parent} with ID: {self.parent.id[1]} and chain: {self.parent.parent}\n")
                pass

        return LPAtoms






