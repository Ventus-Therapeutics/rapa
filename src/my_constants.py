"""
This module contains all the required constants.


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
import numpy as np
from collections import namedtuple

xc_orig=0
yc_orig=0
zc_orig=0

protID = []
out_name = '' 
#rotation for LYS and SER/THR while searching for side chain hydrogen
optRotAngle_LYS = 130
optRotAngle = 360
sp3Angle = 109.5
sp2Angle = 120.0
sp2Angle_rad = 120.0/180.0*np.pi
rad_to_deg = 180.0/np.pi

#search radius
deltaD = 4.0 ##Acceptable distance range!

withinResClashDist = 0.98
btwResClashDist = 1.3

ECutOff = 1.0 ##Degenerate cases

#max num of ASP/GLUS
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
# Note:Counting Proline's N as part of backbone but while assigning behavior-call it neutral
atoms_backbone = np.array([ 'N','CA', 'C', 'O'])###O of water will also be considered if ONLY VALID AA is used


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

#Lone pair for back-bone:
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


