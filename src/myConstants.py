'''All constants for the program '''

import numpy as np
import code
from collections import namedtuple

protID = []
#rotation for LYS and SER/THR while searching for side chain hydrogen
optRotAngle_LYS = 130
optRotAngle = 370

#search radius
deltaD = 4 ##Acceptable distance range!

withinResClashDist = 0.98
btwResClashDist = 1.3

ECutOff = 1.0 ##Degenerate cases
maxCloseAtoms = 10

#max num of ASP/GLUS
numASPs =2
numASPsGLUs =2

#behaviors of donor/acceptor/both/TBD
behavior = namedtuple('behavior',[ 'abbrev', 'IDval']) 
donor = behavior('do', 1)
acceptor = behavior('ac', -1)
doesNothing = behavior('XX', 0)
behavior_dict = {'do':1, 'ac': -1, 'XX': 0, 'TBD':2, 'bo':2}
TBD = behavior('TBD',5)
both = behavior('bo', 2)

#residue of concern/ and residue with OH groups
res_of_concern = np.array(['GLN', 'ASN', 'HIS'])
res_with_OH = np.array(['SER', 'THR', 'TYR'])
# Note:I am counting Proline's N as part of backbone but while assigning behavior I call it neutral 
atoms_backbone = np.array([ 'N','CA', 'C', 'O'])###O water will also come if you dont take ONLY VALID AA 

##List of Atoms that are Acceptors, Donors and all Active atoms. Populate during start of code.
allAcceptors = []
allDonors = []
allActiveAtoms = []
allTBD =[]
allBoth = []
allOnlyAcceptors = []
allOnlyDonors = []
allOnlyDonorsAcceptors = []
allDonorsAcceptorsBoth = []
allUnknowns = []


#List of valid residue names
validResnames =['ALA', 'ARG', 'ASH', 'ASN', 'ASP', 'CYM', 'CYS','CYX','GLH','GLN', 'GLU', 'GLY',
                    'HID','HIE', 'HIP','HIZ','HIM', 'HIS', 'ILE','LEU', 'LYN','LYS', 'MET', 'PHE','PRO','SER','THR','TRP',
                    'TYR','VAL']

#type of HIS/ASN/GLN/ASH/GLH
HIStypes= ['HIP','HIE', 'HID','HIPR', 'HIER', 'HIDR']
HIStypesNoHIP= ['HIE', 'HID', 'HIER', 'HIDR']
ASNtypes = ['ASN', 'ASNR']
GLNtypes = ['GLN', 'GLNR']
ASHtypes = ['ASH_OD1A','ASH_OD2A', 'ASH_OD1B','ASH_OD2B']
GLHtypes = ['GLH_OE1A','GLH_OE2A', 'GLH_OE1B','GLH_OE2B']

##Active atom side chain
HISactiveAtoms = ['ND1', 'NE2']
GLNactiveAtoms = ['OE1','NE2']
ASNactiveAtoms = ['OD1','ND2']
ASHactiveAtoms = ['OD1','OD2']
GLHactiveAtoms = ['OE1','OE2']

#atoms above the active atoms
abvAA_HIS = ['CE1']
abvAA_ASN = ['CG']
abvAA_GLN = ['CD']
abvAA_ASH = ['CG']
abvAA_GLH = ['CD']

#side chain hydrogen atoms attached to HIP/HIE/HID/ASN/GLN/ASH/GLH
HIShAttached = ['HE2']
HIPhAttached = ['HD1','HE2']
HIEhAttached = ['HE2']
HIDhAttached = ['HD1']

ASNhAttached = ['HD21','HD22']
GLNhAttached = ['HE21','HE22']
ASHhAttached = ['HD2']
GLHhAttached = ['HE2']

##creating a dictionary for each residue of concern.
unResDict = {'HIS':[HIStypesNoHIP, HISactiveAtoms, HIShAttached, abvAA_HIS], 
             'GLN':[GLNtypes, GLNactiveAtoms, GLNhAttached, abvAA_GLN], 
             'ASN':[ASNtypes,ASNactiveAtoms, ASNhAttached, abvAA_ASN],
             'ASH':[ASHtypes,ASHactiveAtoms, ASHhAttached, abvAA_ASH],
             'ASP':[ASHtypes,ASHactiveAtoms, ASHhAttached, abvAA_ASH],
             'GLH':[GLHtypes,GLHactiveAtoms, GLHhAttached, abvAA_GLH],
             'GLU':[GLHtypes,GLHactiveAtoms, GLHhAttached, abvAA_GLH]
}

#Hydrogen attached to backbone, and all side chain hydrogen

HBB = [['H']]
HSCnamesARG = [['HE'], ['HH11', 'HH12'], ['HH21','HH22']]
HSCnamesASH = [['HD2']]
HSCnamesGLH = [['HE2']]
HSCnamesASN = [['HD21','HD22']]
HSCnamesGLN = [['HE21','HE22']]
HSCnamesTYR = [['HH']]
HSCnamesTRP = [['HE1']]
HSCnamesHID = [['HD1']]
HSCnamesHIE = [['HE2']]
HSCnamesHIP = [['HD1'],['HE2']]

HSCnamesLYS = [['HZ1','HZ2','HZ3']]
HSCnamesSER = [['HG']]
HSCnamesTHR = [['HG1']]

#Lone pair for back bone:
LPBB = [['LP1', 'LP2']]

#heavys associated with computing side chain lone pairs location of ASP, GLU,ASN, GLN, HIE, HID, SER, THR, TYR, ASH, GLH
hvysForLPsASP = [['OD2', 'OD1','CG'], ['OD1' ,'OD2', 'CG']]
LPSCnamesASP = [['LP3', 'LP4'], ['LP5', 'LP6']]

hvysForLPsGLU = [['OE2', 'OE1','CD'], ['OE1' ,'OE2', 'CG']]
LPSCnamesGLU = [['LP3', 'LP4'], ['LP5', 'LP6']]

hvysForLPsASN = [['ND2', 'OD1','CG']]
LPSCnamesASN = [['LP3', 'LP4']]

hvysForLPsGLN = [['NE2', 'OE1','CD']]
LPSCnamesGLN = [['LP3', 'LP4']]

hvysForLPsHIE = [['CG', 'ND1','CE1']]
LPSCnamesHIE = [['LP3']]


hvysForLPsHID = [['CE1', 'NE2','CD2']]
LPSCnamesHID = [['LP3']]

LPSCnamesSER  = [['LP3']]
LPSCnamesTHR  = [['LP3']]
LPSCnamesTYR  = [['LP3']]

LPSCnamesASH =[['LP3','LP4'],['LP5']]

LPSCnamesGLH =[['LP3','LP4'],['LP5']]


