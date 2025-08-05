"""
This module contains computations related to assessing and assigning states.


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



import os
import sys
import numpy as np

import copy
from collections import Counter
import Bio
from Bio import PDB

import setup_protein as stp
import rapa_residue_atom as rra
import hydrogen_placement_sp2 as hsp2
import hydrogen_placement_sp3 as hsp3
import close_atoms as cats
import global_constants as gc


def setup_HIS(HISres, structure, HIStype = None):

    """
        Objective: To create a new type(HID/HIE/HIP) of HIS in the given structure and 
                   output the original structure with HIS replaced with HID/HIE/HIP

        I/P: HISres: given a HIS residue, structure: where the new HIS type will belong to,
             HIStype: what are you creating either HIE/HIP/HID
        O/P: a dictionary where the structure with new HIE/HIP/HID residues
    """


    dict_struct = {} #Initialize a dictionary to output 
    structName = HIStype 
    dict_struct[structName] = structure
    modelID = HISres.parent.parent.id
    chainID = HISres.parent.id
    resID = HISres.id

    #Setup the name. ##HIStype should be "HIE" or "HID" or "HIP".  by over writing it!!
    dict_struct[structName][modelID][chainID][resID].resname = HIStype
    ####Add the hydrogen
    HIS_dict={'HIP':hsp2.place_hydrogens_HIP, 'HIE':hsp2.place_hydrogens_HIE,'HID':hsp2.place_hydrogens_HID}
    lastSerial = list(dict_struct[structName].get_residues())[-1].child_list[-1].serial_number
    lastSerial, hCoord = HIS_dict[HIStype](dict_struct[structName][modelID][chainID][resID], lastSerial)

    LP_dict = { 'HIE': [gc.hvysForLPsHIE, gc.LPSCnamesHIE], 'HID': [gc.hvysForLPsHID, gc.LPSCnamesHID]}
    try:
        hvys = LP_dict[HIStype][0]
        LPnameAll = LP_dict[HIStype][1]

        hsp2.place_lonepair(dict_struct[structName][modelID][chainID][resID], lastSerial, hvys, LPnameAll)
    except KeyError:
        if gc.log_file: print(f"No Lone pairs to place for HIStype:{HIStype}, and unknown residue: {HISres} on chain: {HISres.parent}")
        pass

    
    return dict_struct[structName]


def setup_rotamer_in_structure(res, structure, placeHyd = True, placeLP = True):
    """
        Objective: To create a rotamer of a given residue in a given structure
        Description: In this function 1.remove any hydrogen/LP and then
                    2.temporarily store original coordinates that need to be swapped
                    3.Add hydrogen and LPS

        I/P: res, and structure where the new rotamer will be placed
        O/P: structure with the residue rotamer present in it
    """

####Basic info regarding residue:
    modelID = res.parent.parent.id
    chainID = res.parent.id
    resID = res.id
########Hydrogen/LP dicts####################################
    HIP_Hyd = ['HD1','HE2']
    HIE_Hyd = ['HE2','LP3']
    HID_Hyd = ['HD1','LP3']
    GLN_Hyd= ['HE21','HE22','LP3','LP4']
    ASN_Hyd= ['HD21','HD22','LP3','LP4']

    removeHdict = {'HIP': HIP_Hyd, 'HIE':HIE_Hyd,'HID':HID_Hyd, 'GLN':GLN_Hyd, 'ASN':ASN_Hyd}
####dictionary of what atoms that are to switch (or the coords need to be switched)
    switchHIS = ['CD2','ND1', 'CE1' , 'NE2']
    switchASN = ['OD1','ND2']
    switchGLN = ['OE1','NE2']
    dictSwitch={'HIS':switchHIS, 'HIP':switchHIS,'HIE':switchHIS, 'HID':switchHIS, 'ASN':switchASN, 'GLN':switchGLN }

    rotomerPlaceH = {'ASN':hsp2.place_hydrogens_ASN, 'GLN': hsp2.place_hydrogens_GLN, 'HID': hsp2.place_hydrogens_HID, 'HIE': hsp2.place_hydrogens_HIE, 'HIP': hsp2.place_hydrogens_HIP}
    
##############################step 1: Remove hydrogen#########################################
    for hyd in removeHdict[res.resname]:
        try:
            res.detach_child(hyd)
        except KeyError:
            if gc.log_file: print(f"Tried to remove hydrogen to setup a rotamer but no hydrogen were "
                                             f"present for: {res} and chain: {res.parent}\n")
##############################step 2: Temp store original atom coords################################
    ##store original
    origAtoms = []
    origAtomsCoords = []
    for count,at in enumerate(dictSwitch[res.resname]):
        origAtoms.append(res[at])
        origAtomsCoords.append(res[at].get_coord())
#############################step3: Swap atom coordinates########################
    resInStruct = structure[modelID][chainID][resID]

###SWAPPING VALS:
    for i in range(len(origAtoms)):
        j=i+1 if(i%2==0) else i-1
        resInStruct[origAtoms[i].id].set_coord(origAtomsCoords[j]  )
    
##############STEP4: If needed: add Hydrogen and LPs
    if(placeHyd):
        lastSerial = list(structure.get_residues())[-1].child_list[-1].serial_number
        try:
            lastSerial, hCoord = rotomerPlaceH[resInStruct.resname](resInStruct, lastSerial)
        except KeyError:
            if gc.log_file: print(f"Tried to place hydrogens to set up a rotamer but no hydrogens for:"
            f" {res} on chain: {res.parent} \n")
            pass

    LP_dict = {'ASP': [gc.hvysForLPsASP, gc.LPSCnamesASP], 'GLU': [gc.hvysForLPsGLU, gc.LPSCnamesGLU],
                'ASN': [gc.hvysForLPsASN, gc.LPSCnamesASN], 'GLN': [gc.hvysForLPsGLN, gc.LPSCnamesGLN],
                'HIE': [gc.hvysForLPsHIE, gc.LPSCnamesHIE], 'HID': [gc.hvysForLPsHID, gc.LPSCnamesHID]
    }


    if(placeLP):
        try:
            hvys = LP_dict[res.resname][0]
            LPnameAll = LP_dict[res.resname][1]
            hsp2.place_lonepair(res, lastSerial, hvys, LPnameAll)
        except KeyError:
            if gc.log_file: print(f"Tried to place lone pairs but no Lone pairs for: {res} on chain:"
            f" {res.parent} \n")
            pass

    resInStruct.isRotamer = 1

    return structure



def create_GLN_ASN_states(unRes, structure):
    """
       Objective: Create GLN rotamer or ASN rotamer configurations
       Input: unknown residue and the structure the residue belongs to
       Output: structDict: A structure with original configuration and
                           rotamer configuration og GLN/ASN
    """

    structDict = {} ##Dict in which output is stored
    mID = unRes.full_id[1]
    cID = unRes.full_id[2]
    unRID = unRes.id[1]
    unfullRID = unRes.id
    unRName = unRes.resname
    
    ##Keep the first struct as the original one
    structDict[ "struct"+str(unRID)+unRName] = structure.copy() 

    #Creating a rotamer and adding it to the dictionary
    structure2rotomer = structure.copy()  
    res2Rotamer = structure2rotomer[mID][cID][unfullRID] 
    structDict["struct"+str(unRID)+unRName+"R"] = setup_rotamer_in_structure(res2Rotamer, structure2rotomer)

    return structDict


def create_HIS_states(unRes, structure):

    """
        Objective: Create all different configs of HIS:
                    1. HID
                    2. HIE
                    3. HIP
                    4. HID_rotomer
                    5. HIE_rotomer
                    6. HIP_rotomer
        Input: the unknown residue of HIS that needs all configs and
            the structure that is of concern
        Output:  all the structs are stores as a list
    """

##This dict will keep the new structures that you will create which will have different configs
    structDict = {}
    skeyList = []
    unRID = unRes.id[1]
    unFullRID = unRes.id

    ##HIS we will create: 
    HISnames = ['HIP', 'HIE','HID']    

    modelID = unRes.parent.parent.id
    chainID = unRes.parent.id
           
    ##Create multiple copies of structures in which HIS resides. We will convert the HIS in the first one to HIP, and HIS in the second one to HIE and HIS in the third one to HID
    structureHIP = structure.copy()    
    structureHIE = structure.copy()    
    structureHID = structure.copy()     
    structuresHIS = [structureHIP, structureHIE, structureHID  ]
    
    ##Setup all the structures with relevant HIStidines:HID/HIE/HIP
    for count,name in enumerate(HISnames):
        skey = "struct"+str(unRID)+name 
        structDict[skey] = setup_HIS(unRes , structuresHIS[count], HIStype = name)
        skeyList.append(skey)
    
    ##Structures with HIP,HIE and HID are formed.
    ###Three structures of HIP, HIE, HID are:  structDict[skeyList[0]], structDict[skeyList[1]] structDict[skeyList[2]]
    ##now rotamer of HIE AND HID is required
    structureHIProtomer = structDict['struct'+str(unRID)+'HIP'].copy()
    structureHIErotomer = structDict['struct'+str(unRID)+'HIE'].copy()
    structureHIDrotomer = structDict['struct'+str(unRID)+'HID'].copy()

    ##rotamer of HIE is needed
    ###hie2 is the HIE residue that you would like to convert to hie2 rotamer
    hip2Rotamer = structureHIProtomer[modelID][chainID][unFullRID]
    hie2Rotamer = structureHIErotomer[modelID][chainID][unFullRID]
    hid2Rotamer = structureHIDrotomer[modelID][chainID][unFullRID]
            
    skey = "struct"+str(unRID)+"HIER" 
    skeyList.append(skey) 
    structDict[skey] = setup_rotamer_in_structure( hie2Rotamer, structureHIErotomer)


    skey = "struct"+str(unRID)+"HIDR"
    skeyList.append(skey)
    structDict[skey] = setup_rotamer_in_structure( hid2Rotamer, structureHIDrotomer)
    skey = "struct"+str(unRID)+"HIPR"
    skeyList.append(skey)
    structDict[skey] = setup_rotamer_in_structure( hip2Rotamer, structureHIProtomer)

    return structDict


def create_ASH(structForASH, ASP_A, ASP_B, LPs):

    """
    objective: to create an ASH with respect to one ASP (APS-A) and one oxygen
              -The oxygen wrt which ASH is created depends on the input LPs
              -If input LPs:[LP3, LP4], then the oxygen we are looking at is OD1
              -If input LPs: [LP5, LP6], then the oxygen we are looking at is OD2

    #Input: -the struct that needs to have ASH
            -ASP_A: ASP where HD2 is going to be attached
            -ASP_B: ASP which is in the h-bonding distance wrt ASP_A
            -LPs: Define which oxygen we are going to be looking at. Also LPs are pseudo atoms,
                 so, we just take advantage of that and replace one of the LPs with HD2

    #Output: the updated structASH where one of the ASPs are converted to ASH!
    """

    

   ## NOTES: Create ASH structure-given two ASPs
    ##the oxygen wrt which HD2 is added depends on the input Lps:
    #If input LPs are:[LP3, LP4] then it is being added to OD1
    ##If input LPs are [LP5, LP5] then it is being added to OD2
    ##compute the minimum distance with respect to each oxygen
    ##Say we are on OD1, OD1 has 2 positions to which HD2 can be added (as it is sp2)
    #compute distance of LP3 wrt resB-OD1, and resB-O2
    #compute distance of LP4 wrt resB-OD1 & res resB-OD2
    #In total we have 4 distance: LP3 wrt (ASP-B-OD1),
    #                             LP3 wrt (ASP-B-OD2)
    #                             LP4 wrt (ASP-B-OD1)
    #                             LP4 wrt (ASP-B-OD2)
   ##Add the hydrogen at that position where the distance is minimum
   ##Newly added hydrogen is attached to OD2!
   ##so incase we were looking at OD1-> we swap it with OD2!!
   ##Add "HD2" on ASP_A
    

    structASH = copy.deepcopy(structForASH)

    #store the original coords of  ASP_A: OD1,OD2,LP3,LP4, LP5, LP6!
    origOD1Coord = ASP_A['OD1'].coord
    origOD2Coord = ASP_A['OD2'].coord

    #bfactor also needs to be switched-as if left unswitched-we have wrong bfactors that are later used!
    origOD1bfactor = ASP_A['OD1'].bfactor
    origOD2bfactor = ASP_A['OD2'].bfactor


    origOD1occupancy = ASP_A['OD1'].occupancy
    origOD2occupancy = ASP_A['OD2'].occupancy
 

    origLP3Coord = ASP_A['LP3'].coord
    origLP4Coord = ASP_A['LP4'].coord
    origLP5Coord = ASP_A['LP5'].coord
    origLP6Coord = ASP_A['LP6'].coord


    modelID_A = ASP_A.parent.parent.id
    chainID_A = ASP_A.parent.id


    modelID_B = ASP_B.parent.parent.id
    chainID_B = ASP_B.parent.id

    ASP_A = structASH[modelID_A][chainID_A][ASP_A.id[1]]
    ASP_B = structASH[modelID_B][chainID_B][ASP_B.id[1]]
    idLP0 = LPs[0]
    idLP1 = LPs[1]
    dist_btw =['ODA_LP3A_OD1B','ODA_LP4A_OD1B','ODA_LP3A_OD2B','ODA_LP4A_OD2B']
    ###Compute all the four distance for the two given lone pairs
    ODA_LP3A_OD1B = ASP_A[idLP0] - ASP_B['OD1']
    ODA_LP4A_OD1B = ASP_A[idLP1] - ASP_B['OD1']

    ODA_LP3A_OD2B = ASP_A[idLP0] - ASP_B['OD2']
    ODA_LP4A_OD2B = ASP_A[idLP1] - ASP_B['OD2']

    list2check = [ ODA_LP3A_OD1B, ODA_LP4A_OD1B, ODA_LP3A_OD2B, ODA_LP4A_OD2B]
    lpList = [ASP_A[idLP0], ASP_A[idLP1], ASP_A[idLP0], ASP_A[idLP1] ] 
    
    ##pick the one which has minimum distance, and the associate lone-pair which we will replace with HD2
    minDist = min(list2check)
    ind = np.where(list2check == minDist)[0][0]
    lpChange = lpList[ind]
    lpChangeCoord = lpChange.coord
    
    if gc.log_file:
        print(f"Need to create ASH from ASP_A:{ASP_A} from {ASP_A.parent}, and {ASP_B} from from {ASP_B.parent} with minimum distance: {minDist}.\n \
        All distances from lone pair(LP3/LP4 or LP5/LP6) associated with OD of ASP_A to OD of ASP_B: {dist_btw} are: {list2check}. Dropping LP:{lpChange} as it is associated with minimum distance \n")

    ##detatch that lone-pair and add HD2 with the lone-pair coordinates
    ASP_A.detach_child(lpChange.name)

    lastSerial = list(structASH.get_residues())[-1].child_list[-1].serial_number
    ASP_A.add(Bio.PDB.Atom.Atom(name='HD2', coord=lpChangeCoord, bfactor=0., occupancy=1., altloc=' ', fullname='HD2', serial_number=lastSerial+1,element='H'))
    structASH[modelID_A][chainID_A][ASP_A.id[1]].resname = 'ASH'


    if(lpChange.id == 'LP4' or lpChange.id == 'LP3'):
            ASP_A.detach_child(ASP_A['OD1'].name)
            ASP_A.detach_child(ASP_A['OD2'].name)

            ASP_A.detach_child(ASP_A['LP5'].name)
            ASP_A.detach_child(ASP_A['LP6'].name)

            if(lpChange.id == 'LP4'):
                ASP_A.detach_child(ASP_A['LP3'].name)
            else:
                ASP_A.detach_child(ASP_A['LP4'].name)

            lastSerial = list(structASH.get_residues())[-1].child_list[-1].serial_number

            ASP_A.add(Bio.PDB.Atom.Atom(name='OD1', coord=origOD2Coord, bfactor = origOD1bfactor, occupancy= origOD1occupancy, altloc=' ', fullname='OD1', serial_number=lastSerial+1,element='O'))
            ASP_A.add(Bio.PDB.Atom.Atom(name='OD2', coord=origOD1Coord, bfactor= origOD2bfactor, occupancy= origOD2occupancy, altloc=' ', fullname='OD2', serial_number=lastSerial+2,element='O'))
            
            
            ASP_A.add(Bio.PDB.Atom.Atom(name='LP3', coord=origLP5Coord, bfactor=0., occupancy=1., altloc=' ', fullname='LP3',serial_number=lastSerial+3,element='LP'))
            ASP_A.add(Bio.PDB.Atom.Atom(name='LP4', coord=origLP6Coord, bfactor=0., occupancy=1., altloc=' ', fullname='LP4',serial_number=lastSerial+4,element='LP'))

            if(lpChange.id =='LP4'):
                ASP_A.add(Bio.PDB.Atom.Atom(name='LP5', coord=origLP3Coord, bfactor=0., occupancy=1., altloc=' ', fullname='LP5',serial_number=lastSerial+5,element='LP'))
            else:
                ASP_A.add(Bio.PDB.Atom.Atom(name='LP5', coord=origLP4Coord, bfactor=0., occupancy=1., altloc=' ', fullname='LP5',serial_number=lastSerial+5,element='LP'))

            return structASH

    elif(lpChange.id == 'LP5'):
            
            ASP_A.detach_child(ASP_A['LP6'].name)
            lastSerial = list(structASH.get_residues())[-1].child_list[-1].serial_number
            ASP_A.add(Bio.PDB.Atom.Atom(name='LP5', coord=origLP6Coord, bfactor=0., occupancy=1., altloc=' ', fullname='LP5', serial_number=lastSerial+1,element='LP')) 
            return structASH
    else:
        return structASH


def create_GLH(structForGLH, GLU_A, GLU_B, LPs):

    """
    objective: to create an GLH with respect to one GLU (GLU-A) and one oxygen
              -The oxygen wrt which GLH is created depends on the input LPs
              -If input LPs:[LP3, LP4], then the oxygen we are looking at is OE1
              -If input LPs: [LP5, LP6], then the oxygen we are looking at is OE2

    #Input: -the struct that needs to have GLH
            -GLU_A: GLU where HE2 is going to be attached
            -GLU_B: GLU which is in the h-bonding distance wrt ASP_B
            -LPs: Define which oxygen we are going to be looking at. Also LPs are added as atoms,
                 so, we just take advantage of that and replace one of the LPs with HE2

    #Output: the updated structGLH where one of the GLUs are converted to GLH!
    """
   ##Notes: compute the minimum distance with respect to each oxygen
   ##Say we are on OE1, OE1 has 2 positions to which HE2 can be added (as it is sp2)
    #compute distance of LP3 wrt resB-OE1, and resB-OE2
    #compute distance of LP4 wrt resB-OE1 & res resB-OE2
    #In total we have 4 distance: LP3 wrt (GLU-B-OE1),
    #                             LP3 wrt (GLU-B-OE2)
    #                             LP4 wrt (GLU-B-OE1)
    #                             LP4 wrt (GLU-B-OE2)
   ##Add the hydrogen at that position where the distance is minimum
   ##Newly added hydrogen is attached to OE2
   ##so in-case we were looking at OE1-> we swap it with OE2
   ##We are adding "HE2" on GLU_A
    
    structGLH = copy.deepcopy(structForGLH)
    #store the original coords of  ASP_A: OD1,OD2,LP3,LP4, LP5, LP6!
    origOE1Coord = GLU_A['OE1'].coord
    origOE2Coord = GLU_A['OE2'].coord
    
    origOE1bfactor = GLU_A['OE1'].bfactor
    origOE2bfactor = GLU_A['OE2'].bfactor


    origOE1occupancy = GLU_A['OE1'].occupancy
    origOE2occupancy = GLU_A['OE2'].occupancy



    origLP3Coord = GLU_A['LP3'].coord
    origLP4Coord = GLU_A['LP4'].coord
    origLP5Coord = GLU_A['LP5'].coord
    origLP6Coord = GLU_A['LP6'].coord

   

    modelID_A = GLU_A.parent.parent.id
    chainID_A = GLU_A.parent.id


    modelID_B = GLU_B.parent.parent.id
    chainID_B = GLU_B.parent.id

    GLU_A = structGLH[modelID_A][chainID_A][GLU_A.id]
    GLU_B = structGLH[modelID_B][chainID_B][GLU_B.id]
    idLP0 = LPs[0]
    idLP1 = LPs[1]
    dist_btw =['OEA_LP3A_OE1B','OEA_LP4A_OE1B','OEA_LP3A_OE2B','OEA_LP4A_OE2B']
    ###Compute all the four distance for the two given lone pairs
    OEA_LP3A_OE1B = GLU_A[idLP0] - GLU_B['OE1']
    OEA_LP4A_OE1B = GLU_A[idLP1] - GLU_B['OE1']

    OEA_LP3A_OE2B = GLU_A[idLP0] - GLU_B['OE2']
    OEA_LP4A_OE2B = GLU_A[idLP1] - GLU_B['OE2']

    list2check = [ OEA_LP3A_OE1B, OEA_LP4A_OE1B, OEA_LP3A_OE2B, OEA_LP4A_OE2B]
    lpList = [GLU_A[idLP0], GLU_A[idLP1], GLU_A[idLP0], GLU_A[idLP1] ] 
    
    ##pick the one which has minimum distance, and the associate lone-pair which we will replace with HD2
    minDist = min(list2check)
    ind = np.where(list2check == minDist)[0][0]
    lpChange = lpList[ind]
    lpChangeCoord = lpChange.coord

    if gc.log_file: print(f"Need to create GLH from GLU_A:{GLU_A} from {GLU_A.parent}, and {GLU_B} from"
    f" {GLU_B.parent} with minimum distance: {minDist}.\n \
    All distances from lone pair(LP3/LP4 or LP5/LP6) associated with OE of GLU_A to OE of GLU_B: {dist_btw} are: {list2check}. Dropping LP:{lpChange} as it is associated with minimum distance \n")

    
    ##detatch that lone-pair and add HE2 with the lone-pair coordinates
    GLU_A.detach_child(lpChange.name)

    lastSerial = list(structGLH.get_residues())[-1].child_list[-1].serial_number
    GLU_A.add(Bio.PDB.Atom.Atom(name='HE2', coord=lpChangeCoord, bfactor=0., occupancy=1., altloc=' ', fullname='HE2', serial_number=lastSerial+1,element='H'))
    structGLH[modelID_A][chainID_A][GLU_A.id].resname = 'GLH'

    #OD2 is bonded to HD2
    ##if LPchange is LP5/LP6 then we are looking at OD2-so no change needed
    #But if LPchange is LP3/LP4: then swap!
    #Also assuring ASH will have lonepair: LP3,LP4,LP5. No LP6 is present

    if(lpChange.id == 'LP4' or lpChange.id == 'LP3'):
            GLU_A.detach_child(GLU_A['OE1'].name)
            GLU_A.detach_child(GLU_A['OE2'].name)

            GLU_A.detach_child(GLU_A['LP5'].name)
            GLU_A.detach_child(GLU_A['LP6'].name)

            if(lpChange.id =='LP4'):
                GLU_A.detach_child(GLU_A['LP3'].name)
            else:
                GLU_A.detach_child(GLU_A['LP4'].name)

            lastSerial = list(structGLH.get_residues())[-1].child_list[-1].serial_number

            GLU_A.add(Bio.PDB.Atom.Atom(name='OE1', coord=origOE2Coord, bfactor = origOE1bfactor, occupancy = origOE1occupancy, altloc=' ', fullname='OE1', serial_number=lastSerial+1,element='O'))
            GLU_A.add(Bio.PDB.Atom.Atom(name='OE2', coord=origOE1Coord, bfactor=origOE2bfactor, occupancy= origOE2occupancy, altloc=' ', fullname='OE2', serial_number=lastSerial+2,element='O'))
            
            
            GLU_A.add(Bio.PDB.Atom.Atom(name='LP3', coord=origLP5Coord, bfactor=0., occupancy=1., altloc=' ', fullname='LP3',serial_number=lastSerial+3,element='LP'))
            GLU_A.add(Bio.PDB.Atom.Atom(name='LP4', coord=origLP6Coord, bfactor=0., occupancy=1., altloc=' ', fullname='LP4',serial_number=lastSerial+4,element='LP'))

            if(lpChange.id =='LP4'):
                GLU_A.add(Bio.PDB.Atom.Atom(name='LP5', coord=origLP3Coord, bfactor=0., occupancy=1., altloc=' ', fullname='LP5',serial_number=lastSerial+5,element='LP'))
            else:
                GLU_A.add(Bio.PDB.Atom.Atom(name='LP5', coord=origLP4Coord, bfactor=0., occupancy=1., altloc=' ', fullname='LP5',serial_number=lastSerial+5,element='LP'))

            return structGLH

    elif(lpChange.id == 'LP5'):
            GLU_A.detach_child(GLU_A['LP6'].name)
            lastSerial = list(structGLH.get_residues())[-1].child_list[-1].serial_number
            GLU_A.add(Bio.PDB.Atom.Atom(name='LP5', coord=origLP6Coord, bfactor=0., occupancy=1., altloc=' ', fullname='LP5', serial_number=lastSerial+1,element='LP')) 
            return structGLH
    else:
        return structGLH


def create_ASH_states(ASP_A_orig, ASP_B_orig, structure):

    """
    objective: To create four different ASP states given two ASP in h-bonding distance with each other
    I/P:
    ASP_A_orig: First ASP in concern
    ASP_B_orig: Second ASP in concern
    structure: structure in concern
    O/P: The four different structures in dict data type
    """

    structForASH = copy.deepcopy(structure)

    modelID_A = ASP_A_orig.parent.parent.id
    modelID_B = ASP_B_orig.parent.parent.id

    chainID_A = ASP_A_orig.parent.id
    chainID_B = ASP_B_orig.parent.id
    
    ASP_A = structForASH[modelID_A][chainID_A][ASP_A_orig.id]
    ASP_B = structForASH[modelID_B][chainID_B][ASP_B_orig.id]

    LPnames = [['LP3', 'LP4'],['LP5','LP6'] ]
    ASP_list = [ASP_A,ASP_A, ASP_B, ASP_B]

    structDict ={}
    sNames =["struct"+ str(ASP_A.id[1]) + "ASH_OD1A",  "struct"+ str(ASP_A.id[1]) + "ASH_OD2A", "struct"+ str(ASP_B.id[1]) + "ASH_OD1B", "struct"+ str(ASP_B.id[1]) + "ASH_OD2B"]
   
    #generating the four states:
    for i in range(1,len(sNames)+1):
        structDict[sNames[i-1]] = create_ASH(structForASH, ASP_list[i-1], ASP_list[-i], LPnames[(i-1)%2])

    return structDict


def create_GLH_states(GLU_A_orig, GLU_B_orig, structure):


    """
    objective: To create four different GLU states given two GLU in h-bonding distance with each other

    I/P: The two different GLUs ID and the structure that needs to be updates
         GLU_A_orig: First ASP in concern
         GLU_B_orig: Second ASP in concern
         structure: structure in concern

    O/P: The four different structures in dict data type
    """
    structForGLH = copy.deepcopy(structure)

    modelID_A = GLU_A_orig.parent.parent.id
    modelID_B = GLU_B_orig.parent.parent.id

    chainID_A = GLU_A_orig.parent.id
    chainID_B = GLU_B_orig.parent.id

    GLU_A = structForGLH[modelID_A][chainID_A][GLU_A_orig.id]
    GLU_B = structForGLH[modelID_B][chainID_B][GLU_B_orig.id]

    LPnames = [['LP3', 'LP4'],['LP5','LP6'] ]
    GLU1 = [GLU_A,GLU_A, GLU_B, GLU_B]

    structDict ={}
    sNames =["struct"+ str(GLU_A.id[1]) + "GLH_OE1A",  "struct"+ str(GLU_A.id[1]) + "GLH_OE2A", "struct"+ str(GLU_B.id[1]) + "GLH_OE1B", "struct"+ str(GLU_B.id[1]) + "GLH_OE2B"]
    

    for i in range(1,len(sNames)+1):
        structDict[sNames[i-1]] = create_GLH(structForGLH, GLU1[i-1], GLU1[-i], LPnames[(i-1)%2])

    return structDict


def get_ASP_GLU_pair(currASP_GLU, structure, ASP_GLUdist):

    """
    objective: For a given ASP/GLU (any of the two oxygen side chain atoms),
               find another ASP/GLU (any of the two oxygen side chain atoms)
               in the h-bonding distance of each other

    input: -currASP_GLU : Given ASP/GLU for which we are looking for another ASP/GLU in its h-bonding distance
           -structure: the structure you want to use to hunt for the other ASP/GLU (you may not need to pass this-just use 
                   currASPs parent?)
           -ASP_GLUdist: the permissible h-bonding distance

    output: the pair ASP_GLU that is in the h-bonding distance within currASP_GLU
    """

    unknownASP_GLUatom = []
    unknownASP_GLUatomInfo = []

    ##Use the list of ALL ASPs to find ASP partner:
    if(currASP_GLU.resname == 'ASP'):
        searchASP_GLUatoms = stp.get_all_unknown_ASP_OD_atoms(structure)
        oxygenName1 = 'OD1'
        oxygenName2 = 'OD2'
    else:
        searchASP_GLUatoms = stp.get_all_unknown_GLU_OE_atoms(structure)
        oxygenName1 = 'OE1'
        oxygenName2 = 'OE2'



    ##remove the OD1, OD2 of Curr ASP as we do not want to include self OD atoms
    searchASP_GLUatoms.remove(currASP_GLU[oxygenName1])
    searchASP_GLUatoms.remove(currASP_GLU[oxygenName2])
    #creating the nbd search object
    ns = Bio.PDB.NeighborSearch(searchASP_GLUatoms)
    
    #search in nbd for any ASP oxygen atom near OD1 of curr ASP and then for OD2
    potUnknownAtom1 = ns.search(currASP_GLU[oxygenName1].coord,ASP_GLUdist)
    potUnknownAtom2 = ns.search(currASP_GLU[oxygenName2].coord,ASP_GLUdist)
   
    #if we find an atom near OD1, append its info
    if(potUnknownAtom1):
        for uat in potUnknownAtom1: 
            unknownASP_GLUatom.append(uat)
            unknownASP_GLUatomInfo.append([currASP_GLU[oxygenName1],currASP_GLU, currASP_GLU.parent, uat, uat.parent, uat.parent.parent, currASP_GLU[oxygenName1]-uat])
    
    #if we find an ASP atom near OD2, we append its info and the atom itself
    if(potUnknownAtom2):
        for uat2 in potUnknownAtom2:
            unknownASP_GLUatom.append(uat2)                
            unknownASP_GLUatomInfo.append([currASP_GLU[oxygenName2],currASP_GLU, currASP_GLU.parent, uat2, uat2.parent, uat2.parent.parent, currASP_GLU[oxygenName2]-uat2])

    
    searchASP_GLUatoms.append(currASP_GLU[oxygenName1])
    searchASP_GLUatoms.append(currASP_GLU[oxygenName2])


    if gc.log_file:
            print(f"Atom found in h-bonding distance neighborhood of {oxygenName1} belonging to {currASP_GLU} of {currASP_GLU.parent} is : {potUnknownAtom1},\n \
                    and  Atom found in h-bonding distance neighborhood of {oxygenName2} belonging to {currASP_GLU} of {currASP_GLU.parent} is : {potUnknownAtom2} \n")
   
    unknownASP_GLUlist = []

    #In the list od unknownASP atom that was just created-find its parent and append it
    for atom in unknownASP_GLUatom:
        unknownASP_GLUlist.append(atom.parent)
    
    #create a set so we can have the unique ASP residues
    unknownASP_GLUset = set(unknownASP_GLUlist)
    #since this code is meant for only diad!

    if(len(unknownASP_GLUset)> gc.numASPsGLUs-1):
        if gc.log_file:
            print(f"number of unique ASPs: {len(unknownASP_GLUset)}. Please check\n\n")
    
    
    if(not unknownASP_GLUset):
        unknownASP_GLUpair = None
    else: 
        unknownASP_GLUpair = list(unknownASP_GLUset)[0]
        
        for unknownASP_GLU in unknownASP_GLUset:
            if gc.log_file:
                print(f"The unknownASP/GLU pair for {currASP_GLU} of {currASP_GLU.parent} is:  {unknownASP_GLU} of "
                     f"chain: {unknownASP_GLU.parent}\n")

    return unknownASP_GLUpair


def energy_of_donor_for_all_close_atoms(hvyAt, allCloseAtoms):
    """
    Objective: To compute energy interaction with neighboring atoms when my reference
                atom(current atom part of the unknown res) is a donor.
        I/P: hvyAt: heavy atom which is the part of unknown residue, allCloseAtoms: All 
                   close atoms we are considering
             con: name of the state, for example: GLN or GLNR or ASH_OD1A
        O/P: enValList: Energy value list, and enSumTotal:  current total sum of the energy
 
    """

    enSumTotal = 0
    enValList = []

    # Add Hydrogen and LP based on name of residue
    struct = hvyAt.parent.parent.parent.parent  ###Since only one struct is being used.

    # Make it a dictionary so it's clear which atom's info is getting used
    pre_cal_donor_info = {'hvyAt': cats.get_info_for_acceptorAt_donorAt(hvyAt)}
    ##Iterating over all close atoms, and implement energy computation depending on close atom behavior 
    #energy computation is done, since heavy atom behavior is already known-Its a donor
    for i in range(1, np.shape(allCloseAtoms)[0]):
        currCloseAtom = allCloseAtoms[i][0]
        closeAtomBehav = rra.my_atom(currCloseAtom).get_behavior().abbrev
        currCloseResID = currCloseAtom.parent.id
        modelIDcloseAt = currCloseAtom.parent.parent.parent.id
        chainIDcloseAt = currCloseAtom.parent.parent.id
        currCloseRes = struct[modelIDcloseAt][chainIDcloseAt][currCloseResID] 

        if(currCloseRes.isSCHknown == 0):
            #This is so that the hvy atom parent is assumed to be known while computing SER close atoms. This flag is turned to 0 and after the computation
            hvyAt.parent.isKnown = 1
            lastSerial = list(struct.get_residues())[-1].child_list[-1].serial_number

            if(currCloseRes.resname=='SER' or currCloseRes.resname=='THR'):
                lastSerial, hLPCoords = hsp3.place_hydrogens_lonepairs_SER_THR(currCloseAtom.parent, lastSerial)
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown = 0
                else:
                    currCloseRes.isSCHknown = 1

                hvyAt.parent.isKnown = 0

            elif(currCloseRes.resname in ['LYS', 'LYN']):
                lastSerial, hLPCoords = hsp3.place_hydrogens_lonepairs_LYS(currCloseAtom.parent, lastSerial)
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown = 0
                else:
                    currCloseRes.isSCHknown = 1
                hvyAt.parent.isKnown = 0
            elif(currCloseRes.resname == 'TYR'):
                lastSerial, hLPCoords = hsp2.place_hydrogens_TYR(currCloseAtom.parent, lastSerial)
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown = 0
                else:
                    currCloseRes.isSCHknown = 1
                hvyAt.parent.isKnown = 0
            else:
                continue
        #After attempting to place hydrogen/LP-if there is no known closeAtoms-then pls continue and try again in another iteration if it comes up!
        if(currCloseRes.isSCHknown==0): 
            if gc.log_file:
                print(f"Current Close atom:{currCloseRes} is still unknown! Going on to the next close atom\n")
            continue
####################################################################################################################
        hvyAt.parent.isKnown = 0
        #often for backbone N-hydrogen is not present
        hPresent=0
        if(hvyAt.id=='N'):
            hPresent=-1
            for at in hvyAt.parent.child_list:
                if(at.id=='H'):
                    hPresent=1
        if(hPresent==-1):            
            if gc.log_file:
                print(f"No backbone hydrogen is present for energy computations. Heavy atom: {hvyAt} belongs to"
                  f" {hvyAt.parent} of its chain: {hvyAt.parent.parent}. Perhaps its the first residue on the chain. \n")
        else:
            atomHs = cats.get_hydrogen_connected_to_donor(hvyAt)
            #going over all the hydrogen atoms connected to the heavy atom
            for hat in atomHs:
                if(closeAtomBehav=='do'):
                    enVal, enSum=cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive=0,
                                                              atype='SP2', hName=hat.id, pre_cal_donor_info=pre_cal_donor_info['hvyAt'])
                    enSumTotal = enSumTotal+enSum
                    enValList.append(enVal)

                elif(closeAtomBehav=='ac'):
                    #Next if it is an acceptor
                    enVal, enSum = cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive=1,
                                                                atype='SP2',hName=hat.id, pre_cal_donor_info=pre_cal_donor_info['hvyAt'])
                    enSumTotal = enSumTotal+enSum
                    enValList.append(enVal)

                elif(closeAtomBehav=='bo'):
                    #If both then-First acceptor
                    enVal0, enSum0 = cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive = 1,
                                                                  atype = 'SP2', hName = hat.id, pre_cal_donor_info=pre_cal_donor_info['hvyAt'])
                    #Next donor
                    enVal1, enSum1 = cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive=0,
                                                                  atype='SP2', hName=hat.id, pre_cal_donor_info=pre_cal_donor_info['hvyAt'])

                    enSumTotal = enSumTotal+enSum0+enSum1
                    enValList.append([enVal0, enVal1])
                    
                else:
                    continue
    return enValList,enSumTotal



def energy_of_acceptor_for_all_close_atoms(hvyAt, allCloseAtoms):

    """
    Objective: To compute energy interaction with neighboring atoms when my reference
               atom(current atom part of the unknown res) is an acceptor.
        I/P: hvyAt: heavy atom which is the part of unknown residue, allCloseAtoms: All
             close atoms we are considering
        O/P: enValList: Energy value list, and enSumTotal:  current total sum of the energy
 
    """

    enSumTotal=0
    enValList=[]
    myHvyAt=rra.my_atom(hvyAt)
    LPAtoms=myHvyAt.get_lonepairs_atoms()

    struct=hvyAt.parent.parent.parent.parent

    pre_cal_acceptor_info = {'hvyAt': cats.get_info_for_acceptorAt_donorAt(hvyAt)}

    for i in range(1, np.shape(allCloseAtoms)[0]):
        currCloseAtom = allCloseAtoms[i][0]
        closeAtomBehav = rra.my_atom(currCloseAtom).get_behavior().abbrev

        currCloseResID = currCloseAtom.parent.id
        modelIDcloseAt = currCloseAtom.parent.parent.parent.id
        chainIDcloseAt = currCloseAtom.parent.parent.id

        currCloseRes = struct[modelIDcloseAt][chainIDcloseAt][currCloseResID]

        #Add the hydrogen if close atom is sp3
        if(currCloseRes.isSCHknown == 0):
            #This is so that the hvy atom parent is assumed to be known while computing SER close atoms. This flag is turned to 0 and after the computation
            hvyAt.parent.isKnown=1
            lastSerial=list(struct.get_residues())[-1].child_list[-1].serial_number
            if(currCloseRes.resname == 'SER' or currCloseRes.resname == 'THR'):
                lastSerial, hLPCoords=hsp3.place_hydrogens_lonepairs_SER_THR(currCloseAtom.parent, lastSerial)
                hvyAt.parent.isKnown=0
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown=0
                else:
                    currCloseRes.isSCHknown=1

            elif(currCloseRes.resname in ['LYS', 'LYN']):
                lastSerial, hLPCoords=hsp3.place_hydrogens_lonepairs_LYS(currCloseAtom.parent, lastSerial)
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown=0
                else:
                    currCloseRes.isSCHknown=1
                hvyAt.parent.isKnown=0

            elif(currCloseRes.resname=='TYR'):
                lastSerial, hLPCoords=hsp2.place_hydrogens_TYR(currCloseAtom.parent, lastSerial)
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown=0
                else:
                    currCloseRes.isSCHknown=1
                hvyAt.parent.isKnown=0

            else:
                continue
    
        #After attempting to place hydrogen/LP-if there is no known closeAtoms-then pls continue and try again in another iteration if it comes up!
        if(currCloseRes.isSCHknown==0): 
            if gc.log_file: print(f"Current Close atom:{currCloseRes} is still unknown! Going on to the "
            f"next close atom \n")
            continue
        hvyAt.parent.isKnown=0
        #Going over all the lone pair atoms for acceptors. All acceptors have lone pairs in-order to accept a H atom
        for j in range(len(LPAtoms)):
                lp_vec = LPAtoms[j].get_vector()
                if(closeAtomBehav == 'do'):
                    #If close atom is a donor
                    enVal,enSum = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 1,
                                                                  atype = 'SP2', pre_cal_acceptor_info=pre_cal_acceptor_info['hvyAt'])
                    #summing the total energy for a given heavy atom
                    enSumTotal = enSumTotal+enSum
                    enValList.append(enVal)                            
                elif(closeAtomBehav == 'ac'):
                    #If close atom is an acceptor
                    enVal, enSum = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 0,
                                                                   atype = 'SP2', pre_cal_acceptor_info=pre_cal_acceptor_info['hvyAt'])
                    #summing the total energy for a given heavy atom
                    enSumTotal=enSumTotal+enSum
                    enValList.append(enVal)

                elif(closeAtomBehav == 'bo'):
                    #If close atom is both then taking the donor first(attractive=1) and then acceptor(attractive=0)
                    enVal0,enSum0 = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 1,
                                                                    atype = 'SP2', pre_cal_acceptor_info=pre_cal_acceptor_info['hvyAt'])
                    enVal1,enSum1 = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 0,
                                                                    atype = 'SP2', pre_cal_acceptor_info=pre_cal_acceptor_info['hvyAt'])

                    enSumTotal=enSumTotal+enSum0+enSum1
                    enValList.append([enVal0, enVal1])
                else:   
                    continue
                

    return enValList,enSumTotal



def compute_energy_for_given_atoms(resState, givenAtoms):

    """
    Objective: Given a configuration, compute energy sum of all the donors and acceptors present in the residue.
    I/P: resState: It is the unknown residue under consideration,
        givenAtoms: atoms we need to consider

    O/P: enSumTotal: Total Energy sum of the configuration, 
         enSumForHvys: list of each heavy atom along with its energy sum
    """

    struct = resState.parent.parent.parent

    enSumTotal = 0
    enSumForHvys = []

   
    #Iterate over the given atoms in the input (as we need the energy for all those atoms with respect to its close atoms)
    for hvyAt in givenAtoms:
        myHvyAt = rra.my_atom(hvyAt)
        myHvyAtBehav = myHvyAt.get_behavior().abbrev

        #create a custom list to find all close atoms
        customList = stp.get_known_donor_acceptor_list_for_one_atom(struct, hvyAt, aaType = 'DONOR_ACCEPTOR_BOTH')
        ##Get a list of all close Atoms (which are donor, acceptor, and both) for the given hvy atom
        allCloseAtoms = cats.get_all_close_atom_info_for_one_atom(hvyAt, customList, dist_cutoff=gc.hbond_d)
        
        if gc.log_file:
            print(f"Evaluate given residue state, hvyAt: {hvyAt} and All close atoms: {allCloseAtoms}")

        if(myHvyAtBehav == 'do'):
            #If the heavy atom is a donor the compute energy for all close atoms
            enValList, enSumTotalDonor = energy_of_donor_for_all_close_atoms(hvyAt, allCloseAtoms)
            enSumTotal = enSumTotal + enSumTotalDonor
            if gc.log_file:
                    print(f"energy sum donor from func:{enSumTotalDonor} and current enSumTotal:{enSumTotal}")
                    print(str(enValList))
        elif(myHvyAtBehav == 'ac'):
            #If heavyAtom is acceptor-compute energy for all its close atom.
            enValList, enSumTotalAcceptor = energy_of_acceptor_for_all_close_atoms(hvyAt, allCloseAtoms)
            enSumTotal = enSumTotal + enSumTotalAcceptor
            if gc.log_file:
                print(f"energy sum acceptor from func:{enSumTotalAcceptor} and current enSumTotal:{enSumTotal}")
                print(str(enValList))
                
        else:
            continue

        enSumForHvys.append([hvyAt, enSumTotal])


    if gc.log_file:
        print(f"For this state:{resState} tot energy is: {enSumTotal}")
        print("************************************************************************")
        print("************************************************************************")

    return enSumTotal, enSumForHvys


def compute_energy_for_given_state(resState):

    """
    Objective: Given a configuration, compute energy sum of all the donors and acceptors present in the residue.
    I/P: config: It is the unknown residue under consideration.
    O/P: enSumTotal: Total Energy sum of the configuration, 
         enSumForHvys: list of each heavy atom along with its energy sum
    """     


    #Find list side chain active atoms, i.e donor/acceptor
    LOAAresState = rra.my_residue(resState).get_unknown_residue_acceptor_donor_atoms()

    #Find the energy for all the side chain active atoms of the unknown residue
    enSumTotal, enSumForHvys = compute_energy_for_given_atoms(resState, LOAAresState)

    return enSumTotal, enSumForHvys


def compute_energy_for_all_states(unknownRes, structStates):

    """
        objective: compute energy for all states of the given unknown residue 
        I/P:
        unknownRes: the unknown residue
        structStates: all the states of unknown residues
        O/P: energy list of all residue stats, and states
    """

    resStates = []
    energyList = []
    #because the modelID and chainID for a given residue-and its different states remain the same (They will be in same structure/chain/model)
    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id

    if(unknownRes.resname == 'ASP' or unknownRes.resname == 'GLU'):
        ##Get the 2 unknownresidue
        sUn = unknownRes.parent.parent.parent
        unknownASPpair = get_ASP_GLU_pair(unknownRes, sUn, gc.deltaD)
        unknownRes0 = [unknownRes, unknownRes, unknownASPpair, unknownASPpair]
    else:
        unknownRes0 = [unknownRes, unknownRes, unknownRes, unknownRes]

    #Going over all states of the unknown residue
    for count,con in enumerate(gc.unResDict[unknownRes.resname][0]):
        #accesing the struct state:
        structState = structStates['struct'+str(unknownRes0[count].id[1])+con]
        #accesing the residue state:
        resState = structState[modelID][chainID][unknownRes0[count].id]
        if gc.log_file:
            print("************************************************************************")
            print(f"State: {con}, with ID {unknownRes0[count].id[1]} and chain: {unknownRes0[count].parent} and isRotamer: {resState.isRotamer}")
        resStates.append(resState)
        #compute energy for the given state:
        enSumTotalResState, enSumForHvys = compute_energy_for_given_state(resState)
        energyList.append([resState, enSumTotalResState, structState, resState, con ])

    return energyList, resStates


def branch_structure(unknownRes, structure):

    """
    objective: to create branch of the structure provided. This function creates all the possible options
               Note: we do not convert it into known at this point as it is only done when setting a state. Here we just create branch to analyze the different rotameric/protonation states.
    Input: -unknownRes-the unknown residue you want to branch
           -structure it belongs to
    Output: S - a dictionary of the multiple structures

    """

    #depending on the name of unknownRes create the structures
    if(unknownRes.resname=='HIS'):      
        S = create_HIS_states(unknownRes, structure)
    elif(unknownRes.resname=='ASN' or unknownRes.resname == 'GLN'):  
        S = create_GLN_ASN_states(unknownRes,structure)
    elif(unknownRes.resname=='ASP' or unknownRes.resname == 'GLU'):
        resPair = get_ASP_GLU_pair(unknownRes, structure, gc.deltaD)
        if(unknownRes.resname=='ASP'):
            S = create_ASH_states(unknownRes, resPair, structure)
        else:
            S = create_GLH_states(unknownRes, resPair, structure)
    else:
        S = None

    return S


def set_state(unknownRes, res2keep, nameOfS2keep, changeVal, branchedS, MSG):

    """
    objective: To set the state in the given structure by converting it to known and removing it from the list of unknowns
        
    Input: -unknownRes: the unknown residue of concern
           -res2keep: the residue state you want to keep
           -nameOfS2keep: name of the structure you want to keep
           -changeVal: number of time changes occur
           -branchedS: the branched structure dict you want to use to assign the value to be known
           -MSG: The message you want to print on screen while setting the state
    
    Output: new and updated structure with known residue, updated list of unknown residue, changeVal and collect Data
    """
    struct = branchedS[nameOfS2keep]
    
    if gc.log_file:
        print(MSG)

    changeVal+=1
    
    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id
    struct[modelID][chainID][unknownRes.id].isKnown = 1

    if(unknownRes.resname == 'ASP' or unknownRes.resname == 'GLU'):
        sUn = unknownRes.parent.parent.parent
        unknownASP_GLUpair = get_ASP_GLU_pair(unknownRes, sUn, gc.deltaD)
        modelIDpair = unknownASP_GLUpair.parent.parent.id
        chainIDpair = unknownASP_GLUpair.parent.id
        struct[modelIDpair][chainIDpair][unknownASP_GLUpair.id].isKnown = 1

    structure = copy.deepcopy(branchedS[nameOfS2keep])

    return structure, changeVal


def get_structure_name(res, moreInfo=None):
    
    """
    objective: to get structure name-specifically by checking if the unknown residue it is a rotamer or not

    Input:  -res: is the residue to check
            -moreInfo: name of res describing the unknown res. For example, "HIER" is HIE in rotameric state(non orig state).
            It will describe the name of structure as 'struct123HIER' assuming 123 is residue number of HIE in consideration.
    Output: nameOfStruct: -name of the struct
    
    """
    nameOfStruct = 'struct'+str(res.id[1]) + moreInfo

    return nameOfStruct


def get_degenerate_structure_names(degenArray):

    """
    objective: To extract all the structure names for the degenerate cases using the dgenArray data type
    Input: degenArray:- It is the data type which contains information for the degenCases
    Output: Names of all structs that is extracted from degenArray
    """

    #list of degen struct names for output
    degenStructNames = []
    ##degenArray has 2 columns. Zeroth column:residue and first column has energy value
    ##Iterating over and extracting the names by using get_structure_name
    for count, res in enumerate(degenArray[:,0]): 
        nameOfStruct = get_structure_name(res, moreInfo = degenArray[count,-1])
        degenStructNames.append(nameOfStruct)
    if gc.debug:
        print(f"The degenerate structure names are: {degenStructNames}\n And more details are(resState, "
              f"enSumTotalResState, structState, resState, con/name of the state ): {degenArray}\n")

    return degenStructNames


def get_HIP_energies(resHIP, resHIPR):

    """
    objective:-get energy for the HIP cases (original state + rotomer)
              -The two atoms: ND1 and NE2 needs to be treated separately as interaction energies
                              with both atoms require to be negetive for HIS to be a HIP case!

    Input: -resHIP: the original state of HIP
           -resHIPR: the rotamer state of HIP

    Output: -enSumTotND1: energy for ND1 atom of HIP original state
            -enSumTotNE2: energy for NE2 atom of HIP original state
            -enSumTotND1Ro:  energy for ND1 atom of HIP rotomer state
            -enSumTotNE2Ro: energy for NE2 atom of HIP rotomer state

    """
    enSumTotND1, enSumForND1 = compute_energy_for_given_atoms(resHIP, [resHIP['ND1']])
    enSumTotNE2, enSumForNE2 = compute_energy_for_given_atoms(resHIP, [resHIP['NE2']])

    enSumTotND1Ro, enSumForND1Ro = compute_energy_for_given_atoms(resHIPR, [resHIPR['ND1']])
    enSumTotNE2Ro, enSumForNE2Ro = compute_energy_for_given_atoms(resHIPR, [resHIPR['NE2']])

    return enSumTotND1, enSumTotNE2, enSumTotND1Ro, enSumTotNE2Ro



def evaluate_degenerate_cases(unknownRes, structure, S, energyArray, sortedEnArray, changeVal, skipVal, skipResInfo):


    """
    objective: to update structures for the potential degenerate cases. By def: a degenerate case is
                where the two or more different states of a residue have energy difference less than 1 Kcal.
    
    Input: -unknownRes: the unknown residue with respect to which the degenerate case is developed
           -structure: the structure that needs to be updated.
           -S: it is a dictionary of multiple structures of the multiple degenerate cases.
           -LOCA_unknownRes: List of Close Atoms of the unknown residue
           -energyArray: energy  information for all the degenerate cases
           -sortedEnArray: energy information sorted in the min to max order
           -changeVal: change value of the unknown->known residue needs to be tracked as degenerate
                        cases are fixed up
           -skipVal:skip value of the unknown->unknown residue(no change) needs to be tracked as well
           -skipResInfo: collecting information where no changes are made-and the unknown residue is skipped


    Output:
           -structure: new and updated structure (If it is a truly degenerate case then structure remains the same as input structure)
           -changeVal: keeping track of changes that occur(unknown->known residues)
           -skipVal: keep track  of residues that are skipped in the list of unknowns
           -skipResInfo: updated info if the unknown residue's state is not yet set.This includes: unknown redisue, degenerate info, degenerate structure names, structure associated.
    """

    ##Evaluating 5 possible cases
    #1. all energy values are zero
    #2. all energy values are pos
    #3. all energy values are negative
    #4. smallest energy value is zero
    #5. smallest energy value is negative-not degenerate if the second-smallest energy value=0 or >0

    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id

    sortedResNamesArr = np.array([x.resname for x in sortedEnArray[:,0]])
    ##Create conditionals for all possibilities
    allEnergyZero = all(c == 0 for c in energyArray[:,1])    ##All close atoms have zero energy
    allEnergyPos = all(c > 0 for c in energyArray[:,1])
    allEnergyNeg = all(c < 0 for c in energyArray[:,1])
    smallestEnergyZero = sortedEnArray[0,1]==0
    smallestEnergyNeg =sortedEnArray[0,1]<0


    if(allEnergyZero):
    ##not degenerate
        #IF HIS->default to HIE else keep the original state
        if(unknownRes.resname=='HIS'): 
            nameOfRes2keep='HIE'
        else:
            nameOfRes2keep=unknownRes.resname

        if(unknownRes.resname=='GLU' or unknownRes.resname=='ASP'):
            del S
            sUn = unknownRes.parent.parent.parent
            unknownASP_GLUpair = get_ASP_GLU_pair(unknownRes, sUn, gc.deltaD)
            modelID_pair = unknownASP_GLUpair.parent.parent.id
            chainID_pair = unknownASP_GLUpair.parent.id


            structure[modelID][chainID][unknownRes.id].isKnown=1
            structure[modelID_pair][chainID_pair][unknownASP_GLUpair.id].isKnown=1
            res2keep=unknownRes 
            MSG=f'{unknownRes} and pair: {unknownASP_GLUpair} All energies = 0  '
            if gc.log_file:
                print(MSG)
            changeVal +=1
            return structure, changeVal, skipVal, skipResInfo

        nameOfS2keep = 'struct'+str(unknownRes.id[1])+nameOfRes2keep
        res2keep= S[nameOfS2keep][modelID][chainID][unknownRes.id]
        
        #min energy stored is the energy value picked depending on the residue name to keep
        ind = np.where(sortedResNamesArr[:]== res2keep.resname)[0][0]
        enValPick =sortedEnArray[ind][1]
       
        #string that will be printed out to the user along with other default info.
        condStr = "ALL ENERGIES = 0"

        msg2Usr = f"{condStr}, Saving residue: {res2keep} to residue state corresponding to minimum energy  i.e {enValPick} and isRotamer:{res2keep.isRotamer}"
        ###Setting the state-change value and skip values are updated in the function (set_state)
        structure, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, changeVal, S, msg2Usr)

        del S

    elif(smallestEnergyZero):
        #not degenerate
         res2keep = sortedEnArray[0,0] ##PICKING the smallest as 0<+v, -v<0

         moreInfoName = sortedEnArray[0,-1]
         nameOfS2keep = get_structure_name(res2keep, moreInfo = moreInfoName)
 
         #Min energy is the smallest in the sorted row 
         enValPick =sortedEnArray[0][1]  
         #string that will be printed out to the user along with other default info.
         condStr = "Smallest Energy is zero and the next one is either pos or zero!"
         msg2Usr = f"{condStr},Saving residue: {res2keep} to residue state corresponding to minimum energy  i.e {enValPick} and isRotamer:{res2keep.isRotamer} (specifically:smallest: {sortedEnArray[0,1]} and next smallest:{sortedEnArray[1,1]})"
         ###Setting the state-change value and skip values are updated in the function (set_state)
         structure, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, changeVal, S, msg2Usr)

         del S

    elif(smallestEnergyNeg):
        # not a degenerate case if the second smallest is not zero or >0
        if(sortedEnArray[1,1]==0 or sortedEnArray[1,1]>0):
            #not degenerate case
            res2keep = sortedEnArray[0,0]

            moreInfoName = sortedEnArray[0,-1]
            nameOfS2keep = get_structure_name(res2keep, moreInfo = moreInfoName)
            #Min energy is the smallest in the sorted row 
            enValPick =sortedEnArray[0][1]
           
            #string that will be printed out to the user along with other default info.
            condStr = "Smallest Energy is negative and the next positive or zero!"

            msg2Usr = f"{condStr}, Saving residue: {res2keep} to residue state corresponding to minimum energy  i.e {enValPick} and isRotamer:{res2keep.isRotamer}, (specifically:smallest: {sortedEnArray[0,1]} and next smallest:{sortedEnArray[1,1]})"
            ###Setting the state-change value and skip values are updated in the function (set_state)
            structure, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, changeVal, S, msg2Usr)

            del S
        else:
            #A degenerate: as smallest and the second-smallest energies are negative
            degenList = []
            #find out how many negetive energy values are present and append it to the degenList
            for count,sEn in enumerate(sortedEnArray[:,1]):
                if(sEn < 0):   
                    degenList.append(sortedEnArray[count,:])
                else:
                    continue
            degenArray = np.array(degenList)
            MinEn = sortedEnArray[0,1]
    
            #find the difference with minimum energy
            EnDiffWithMin =abs(degenArray[:,1]-MinEn)
            #get the index where difference of energy is less than 1 kcal
            indDegen = np.where(EnDiffWithMin<gc.ECutOff)
            #use the index where diff of energy is less than 1 kcal to form a degenInfo
            degenInfo = degenArray[indDegen,:]
            a,b,c =degenInfo.shape
            ##c=2 always(as it is res, energyVal)#b= number of degen cases#a=1
            degenInfo = degenInfo.reshape(a*b,c)

            skipVal+=1
            #store the struct name
            degenStructNames = get_degenerate_structure_names(degenInfo)
            #update skip info
            skipResInfo.append([unknownRes, degenInfo,degenStructNames,S])
            if gc.log_file:
                print(f"Skipping: {structure[modelID][chainID][unknownRes.id[1]]} as lowest vals:"
                  f"{sortedEnArray[0,1]} and {sortedEnArray[1,1]}, degenergate Info: {degenInfo} with {EnDiffWithMin[indDegen]}<{gc.ECutOff}")
                print("###########################################################################")


    elif(allEnergyPos or allEnergyNeg):
            ##Degenerate case
            MinEn = sortedEnArray[0,1]

            if gc.log_file:
                print(f"All Energy Values are positive OR negative: {sortedEnArray}")

            #find the difference with minimum energy
            EnDiffWithMin =abs(sortedEnArray[:,1]-MinEn)
            #get the index where difference of energy is less than 1 kcal
            indDegen = np.where(EnDiffWithMin<gc.ECutOff)
            #use the index where diff of energy is less than 1 kcal to get degenNames, using the sortedResNamesArr
            degenNames = sortedResNamesArr[indDegen]
            #create the degenArray using the index where diff of energy<1Kcal in the sortedEnArray
            degenArray = sortedEnArray[indDegen]

            skipVal+=1
            #getting the degenerate structure names
            degenStructNames = get_degenerate_structure_names(degenArray)
            #updating skipResInfo
            skipResInfo.append([unknownRes, degenArray,degenStructNames,S])

            if gc.log_file:
                print(f"Please Skip: {structure[modelID][chainID][unknownRes.id[1]]} as lowest vals:{sortedEnArray[0,1]} and {sortedEnArray[1,1]}, degenergate names: {degenNames} and values are: {degenArray} with {EnDiffWithMin}<{gc.ECutOff}")
                print("###########################################################################")
                
    else:
           if gc.log_file:
               print("ERROR: SHOULD NOT GO THROUGH DEGENERATE CASES!")
           sys.exit()
           
    
    return structure, changeVal, skipVal, skipResInfo


def evaluate_HIP_cases(unknownRes, structure, S, changeVal, skipVal, skipResInfo) :
    """
    objective: To investigate the HIS could be HIP/HIP rotamer
    I/P:-unknownRes: the unknown HIS to investigate
        -structure: structure to update
        -S: is a dictionary of possible structures corresponding to HIP/HIE/HID and rotamers
        -changeVal: change value of the unknown->known residue needs to be tracked as degenerate
                        cases are fixed up
        -skipVal:skip value of the unknown->unknown residue(no change) needs to be tracked as well
        -skipResInfo: collecting information where no changes are made-and the unknown residue is skipped.This includes: unknown residue, degenerate info, degenerate structure names, structure associated

    O/P:
        -structure: new and updated structure
        -changeVal: keeping track of changes that occur(unknown->known residues)
        -skipVal: keep track  of residues that are skipped in the list of unknowns
        -skipResInfo: updated info if the unknown residue's state is not yet set!
        -S:update the dictionary structure
        -HIPset: if HIS is set as HIP-> then set this flag to 1/true
        -HIPdegen: if HIS is a degenerate case between HIP and HIP rotamer then set this flag to true!!

    """

    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id

    sHIP = 'struct'+str(unknownRes.id[1])+'HIP' 
    resHIP = S[sHIP][modelID][chainID][unknownRes.id]

    sHIPR = 'struct'+str(unknownRes.id[1])+'HIPR' 
    resHIPR = S[sHIPR][modelID][chainID][unknownRes.id]
    HIPset = 0
    HIPdegen = 0


    enSumTotND1, enSumTotNE2, enSumTotND1Ro, enSumTotNE2Ro   = get_HIP_energies(resHIP, resHIPR)
    sumOrigState = enSumTotND1 + enSumTotNE2
    
    sumRotamer = enSumTotND1Ro + enSumTotNE2Ro
    origStateHIP = enSumTotND1< -1*gc.ECutOff and enSumTotNE2< -1*gc.ECutOff
    rotamerStateHIP = enSumTotND1Ro< -1*gc.ECutOff and enSumTotNE2Ro<-1*gc.ECutOff

    #Checking the degenerate case
    if(origStateHIP and rotamerStateHIP and abs(sumOrigState - sumRotamer)<1*gc.ECutOff and (enSumTotND1 !=0 and enSumTotNE2 !=0 and enSumTotND1Ro !=0 and enSumTotNE2Ro !=0 ) ):
          if gc.log_file:
              print(f"HIP:{unknownRes} of {unknownRes.parent} is DEGENERATE!!")

          #create the list and array of energy information
          energyList = [[resHIP, sumOrigState,S[sHIP], resHIP,'HIP'], [resHIPR, sumRotamer, S[sHIP],resHIPR ,'HIPR']]
          energyArray = np.array(energyList)
          sortedEnList = sorted(energyList,key=lambda x: (x[1]))
          sortedEnArray = np.array(sortedEnList)


          #check the degenerate cases
          skipValb4check = skipVal
          structure, changeVal, skipVal, skipResInfo = evaluate_degenerate_cases( unknownRes, structure, S,
                                                                                  energyArray, sortedEnArray, changeVal, skipVal, skipResInfo)
          if(abs(skipValb4check - skipVal)==0):
              HIPdegen =0
          else:
              HIPdegen = 1
              del S['struct'+str(unknownRes.id[1])+'HIE' ]
              del S['struct'+str(unknownRes.id[1])+'HIER' ]
              del S['struct'+str(unknownRes.id[1])+'HID' ]
              del S['struct'+str(unknownRes.id[1])+'HIDR' ]

          return structure, changeVal, skipVal, skipResInfo, S, HIPset, HIPdegen

    elif(enSumTotND1< -1*gc.ECutOff and enSumTotNE2< -1*gc.ECutOff):
        ##check if the original state is a possibility
        ##Also check if the closest atom is OG from SER or OG1 from THR. If it is then HIP cannot be the state.
        ##Because OG/OG1 are poor acceptors
        if gc.log_file:
            print(f"possibility of original state HIP as the two energy values are: {enSumTotND1} and {enSumTotNE2}")
        myHIP = rra.my_residue(resHIP)
        LOAA_unknownRes = myHIP.get_unknown_residue_acceptor_donor_atoms()
        
###############################Check if one of the acceptor position of SER/THR#####################################
        ##Get the list of close atoms for the list of active atoms.This list of close atoms may not contain only known atoms
        LOCA_unknownRes, dim = cats.get_list_of_close_atoms_for_list_of_atoms(LOAA_unknownRes,'DONOR_ACCEPTOR_BOTH')
    
        ##dim[0] id number of close atoms for first nitrogen and dim[1] is num of close atoms for second nitrogen.
        ##if there is no interaction with either of the  atoms-then there should NOT be considered a HIP at all since both interactions must be negative!!
        if(dim[0]>1 or dim[1]>1):
            #There are two lists created. One will be for ND1, and other will be for NE2.
            Nlist1 = LOCA_unknownRes[0]
            Nlist2 = LOCA_unknownRes[1]
            
            ##Create an array and a list of close atom names to iterate over and find if one of the close atoms is OG1/OG or SER/THR
            closeAtsArr1 = np.array(Nlist1)[:,0]
            closeAtsArr2 = np.array(Nlist2)[:,0]

            ###The first atom is the reference atom for HIS (ND1/NE2). Both act as a donor in the case of HIP!
            closeAtsArr = [closeAtsArr1, closeAtsArr2]
            cannotBeHIP = []
        
            ##Iterate over the two close atom array, and over each atom in that array to look for OG/OG1.
            #If found append to cannot be HIP
            for cAarray in closeAtsArr:
                for atom in cAarray:
                    if((atom.name=='OG' and atom.parent.resname == 'SER') or (atom.name=='OG1' and atom.parent.resname == 'THR')):
                        cannotBeHIP.append(1)

            if(any(a == 1 for a in cannotBeHIP)):
                ##If it cannot be HIP-delete the appropriate structures and move on
                if gc.log_file:
                    print(f"cannotBeHIP: {cannotBeHIP}")
                del S['struct'+str(unknownRes.id[1])+'HIP' ]
                del S['struct'+str(unknownRes.id[1])+'HIPR' ]

            else:
                res2keep = resHIP
                nameOfS2keep = sHIP
                saveText = f"Saving residue: {res2keep.resname} to residue state corresponding to minimum energy i.e ND1Ro: {enSumTotND1} + NE2Ro: {enSumTotNE2} isRotamer:{res2keep.isRotamer}"
                structure, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, changeVal, S, saveText)
                HIPset = 1 

                del S['struct'+str(unknownRes.id[1])+'HIE' ]
                del S['struct'+str(unknownRes.id[1])+'HIER' ]
                del S['struct'+str(unknownRes.id[1])+'HID' ]
                del S['struct'+str(unknownRes.id[1])+'HIDR' ]
                del S['struct'+str(unknownRes.id[1])+'HIPR' ]
        
        else: 
            if gc.log_file:
                print(f"ERROR: Investigating HIP-where energies for both nitrogen's<-{1*gc.ECutOff} but close atoms = 0. PLS CHECK CODE ")
            sys.exit()

    elif(enSumTotND1Ro< -1*gc.ECutOff and enSumTotNE2Ro<-1*gc.ECutOff):
        ##check if the rotamer state is a possibility
        ##Check if the closest atom is OG from SER or OG1 from THR. If it is then HIP cannot be the state
        ##Because OG/OG1 are poor acceptors
        if gc.log_file:
            print("possibility of rotamer state HIP...")
        myHIPR = rra.my_residue(resHIPR)

        LOAA_unknownResR = myHIPR.get_unknown_residue_acceptor_donor_atoms()

###############################Check if one of the acceptor position is SER/THR#################################################
        #Create the list of close atoms for active atoms
        #This list of close atoms may not contain only known atoms
        LOCA_unknownResR, dim = cats.get_list_of_close_atoms_for_list_of_atoms(LOAA_unknownResR,'DONOR_ACCEPTOR_BOTH')
        ##if there is no interaction with either of the atoms-then it should not be considered a HIP at all since both interactions must be negative for HIP to occur
        if(dim[0]>1 or dim[1]>1):
            #There are two lists created. One will be for ND1, and other will be for NE2.
            Nlist1 = LOCA_unknownResR[0]
            Nlist2 = LOCA_unknownResR[1]
            
            ##Create an array and a list of close atom names to iterate over and find if one of the close atoms is OG1/OG or SER/THR
            closeAtsArr1 = np.array(Nlist1)[:,0]
            closeAtsArr2 = np.array(Nlist2)[:,0]

            ####The first atom is the reference atom for HIS (ND1/NE2). Both act as a donor in the case of HIP.
            closeAtsArr = [closeAtsArr1, closeAtsArr2]
            
            cannotBeHIPR = []
            
            ##Iterate over the two close atom array, and over each atom in that array to look for OG/OG1.
            #If found append to cannot be HIP
            for cAarray in closeAtsArr:
                for atom in cAarray:
                    if((atom.name=='OG' and atom.parent.resname == 'SER') or (atom.name=='OG1' and atom.parent.resname == 'THR')): 
                        cannotBeHIPR.append(1)

            if(any(value == 1 for value in cannotBeHIPR)):
                ##If it cannot by HIP-delete the appropriate structures.
                if gc.log_file:
                    print(f"cannotBeHIPR: {cannotBeHIPR}")

                del S['struct'+str(unknownRes.id[1])+'HIP' ]
                del S['struct'+str(unknownRes.id[1])+'HIPR' ]

            else: 
                #It is HIP rotamer:
                res2keep = resHIPR
                nameOfS2keep = sHIPR
                saveText = f"Saving residue: {res2keep.resname} to residue state corresponding to minimum energy i.e ND1Ro: {enSumTotND1Ro} + NE2Ro: {enSumTotNE2Ro} isRotamer:{res2keep.isRotamer}"
                structure, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, changeVal, S, saveText)
                HIPset = 1 

                del S['struct'+str(unknownRes.id[1])+'HIE' ]
                del S['struct'+str(unknownRes.id[1])+'HIER' ]
                del S['struct'+str(unknownRes.id[1])+'HID' ]
                del S['struct'+str(unknownRes.id[1])+'HIDR' ]
                del S['struct'+str(unknownRes.id[1])+'HIP' ]

        else: 
            if gc.log_file:
                print(f"Investigating HIP Rotamer-where energies for both nitrogen's<-{1*gc.ECutOff} but close atoms = 0. Check more details.")
            sys.exit()

    else:
        del S['struct'+str(unknownRes.id[1])+'HIP' ]
        del S['struct'+str(unknownRes.id[1])+'HIPR' ]

    return structure, changeVal, skipVal, skipResInfo, S, HIPset, HIPdegen


def iterate_list_of_unknown_residues_and_set_states(structure):
    """
    objective: To iterate over a given structure and to set states depending on energy computations
    I/P: structure: structure of a protein

    O/P: structure: updated structure
         changeVal: number of changes that occur
         skipVal: num of residues skipped
         skipResInfo: collecting information where no changes are made-and the unknown residue is skipped. This includes: unknown residue, degenerate info, degenerate structure names, structure associated
    """

    skipVal = 0
    changeVal =0 

    skipResInfo = []

    unknownResIter = stp.get_unknown_residue_list(structure)

    lenUnResOrig = len(unknownResIter)

    unknownResNames = []
    for res in unknownResIter:
        unknownResNames.append(res.resname)

    uniqRes = Counter(unknownResNames).items()

    if gc.log_file:
        print("\n\nNew iterate_list_of_unkwown_residues call")
        print(f"Unknown residue to iterate over: {unknownResIter}, and its length: {len(unknownResIter)}")
        print(f"The unknowns present: {uniqRes}\n\n")

    
    
    for count, unknownResOrig in enumerate(unknownResIter):    
        modelID = unknownResOrig.parent.parent.id
        chainID = unknownResOrig.parent.id
        unknownRes = structure[modelID][chainID][unknownResOrig.id]
        if(unknownRes.isKnown == 1):continue
        
        if gc.log_file:
            print("#"*160)
            print(f"unknown residue number: {count+1}/{lenUnResOrig} and unknown res is: {unknownRes} and its known val:{unknownRes.isKnown} and is rotamer:{unknownRes.isRotamer}, Skip value:{skipVal}, ChangeVal: {changeVal}")


        current_unknown_res = rra.my_residue(unknownRes)
        #Get list of active atoms, list of close atoms, and number of close atoms
        LOAA_unknownRes = current_unknown_res.get_unknown_residue_acceptor_donor_atoms()
        ##This adds close points for rotamer HIS as well:
        if unknownRes.resname == 'HIS':
            LOAA_unknownRes.append(structure[modelID][chainID][unknownRes.id]['CD2'])
            LOAA_unknownRes.append(structure[modelID][chainID][unknownRes.id]['CE1'])


        # when we check for if there are polar atoms nearby an unknown residue, it has to be atom based
        # even in an unknown residue, the backbone atom of that unknown residue should still be treated as known
        # when we populate the neighbor list, we only populate for unknown side chain atoms in that residue and not
        # every atom in that residue


        #Get all the close atoms for the atoms in concern
        #This list will have all polar atoms (whether known or unknown)
        LOCA_unknownRes, dim = cats.get_list_of_close_atoms_for_list_of_atoms(LOAA_unknownRes,
                                                                               'DONOR_ACCEPTOR_BOTH_TBD')
        #we need to get the atoms in LOAA_unknownRes(that's the unknown pair we're trying to look at right now) out
        # of the neighbor list
        total_valid_neighbor_count = 0 # total number of neighbor polar atoms (known and unknown included)
        total_known_neighbor_count = 0 # total number of neighbor known polar atoms
        for neighbor_list in LOCA_unknownRes:
            for i in neighbor_list:
                # first exclude atoms in LOAA_unknownRes
                if True not in [rra.if_two_atoms_are_same(i[0], _) for _ in LOAA_unknownRes]:
                    total_valid_neighbor_count += 1
                    # an atom can be known either their parent residue is known or they're backbone atoms
                    if i[0].parent.isKnown or rra.my_atom(i[0]).is_backbone():
                        total_known_neighbor_count += 1

        # For HIS, ASN, GLN, if there are no polar atoms (unknown residues included), if it's HIS, set HIS-> HIE,
        # if ASN/GLN keep them as they are and mark them known
        # All the other residues, they will remain unknown
        if(total_valid_neighbor_count == 0):

            if unknownRes.resname in ['HIS', 'ASN', 'GLN']:
                if gc.log_file:
                    print(f"This is a {unknownRes.resname} w/o polar atoms nearby, setting state to known!")

                if unknownRes.resname == 'HIS':
                    setup_HIS(unknownRes, structure, 'HIE')
                structure[modelID][chainID][unknownRes.id].isKnown = 1
                unknownRes.isKnown = 1
                changeVal += 1
            # for other residues, do nothing, they would remain unknown
            if gc.log_file:
                print(f"There are no known residues nearby for current residue: {unknownRes.resname},leave it as unknown...")

            continue

        else:
            # if there are no known polar atoms nearby, do nothing and leave it as unknown (we need at least one
            # known polar atoms nearby to evaluate the states)
            if total_known_neighbor_count == 0:
                if gc.log_file:
                    print("There are no known residues nearby for current residue: {unknownRes.resname},leave it as unknown...")
                continue

            if gc.log_file:
                print(f"There are known residues nearby for current residue: {unknownRes.resname},branching the structure to check the energy!!\n")

            # if there are known nearby, branch the structure and keep the one with lowest energy
            #create branch of a given unknown residue.
            S = branch_structure(unknownRes, structure)

            energyList, resStates = compute_energy_for_all_states(unknownRes, S)

            energyArray = np.array(energyList, dtype=object)
            sortedEnList = sorted(energyList,key=lambda x: (x[1]))
            sortedEnArray = np.array(sortedEnList, dtype=object)

            # sortedEnArray: column 0: residue objects, column1: energy value
            allEnergyZero = all(c == 0 for c in energyArray[:,1])    ##All close atoms have zero energy

            if(unknownRes.resname == 'HIS' and not(allEnergyZero)): 
                #If HIS and all energy values are not zero - test the possibility of HIP:
                structure, changeVal, skipVal, skipResInfo, S, HIPset, HIPdegen = evaluate_HIP_cases(
                    unknownRes, structure, S, changeVal, skipVal, skipResInfo)

                if(HIPdegen==1 or HIPset==1):
                    if gc.log_file:
                        print("Continuing to the next unknown in the for loop as HIP is set or HIP is degenerate")
                    continue

                if gc.log_file:
                    print("continuing as HIP is not an option!!")

            else:
                #If not a HIS/all HIS energy values are zero. Declare HIP is not possible.
                if gc.log_file:
                    print(f"Not checking for HIP- as I am {unknownRes} of {unknownRes.parent} and currently am not on HIS or all the HIS energies are zero\n")
    #########Take the smallest value and subtract from all other values. Find out how many values are<1
            MinEn = sortedEnArray[0,1]
            #The first value is the minimum. So starting from second Val
            diffWithMin = abs(sortedEnArray[1:,1]-MinEn)

            ##checking if the difference between the smallest two energy value is less than 1-thus a possible degeneracy!
            degenFound = any(diffWithMin< gc.ECutOff)
###################################IF DEGENERATE ###############################################################
            if(degenFound):
                if gc.log_file:
                    print(f"For current residue: {unknownRes.resname} there are more than one states whose energy "
                          f"difference is less than 1kcal\nSorted energy for all states are: {sortedEnArray[:,1]}\nStart evaluating if they are actually degenerate...")

                #if smallest two energies are less than ECutOff then explore the possibility of degeneracy
                # if the states are actually degenerate, leave the residue as unknown
                # the degenerate state info is kept in skipResInfo
                structure, changeVal, skipVal, skipResInfo = evaluate_degenerate_cases(unknownRes, structure, S, energyArray, sortedEnArray, changeVal, skipVal, skipResInfo)
            else:
                ##If not degenerate, pick the state with the smallest energy value and set it as known
                res2keep = sortedEnArray[0,0] ##Picking the smallest
                ind = 0
                moreInfoName = sortedEnArray[0,-1]
                nameOfS2keep = get_structure_name(res2keep, moreInfo = moreInfoName)

                enValPick =sortedEnArray[ind][1] # originally used for printing
                if  gc.log_file:
                    print(f"One state for current residue: {res2keep.resname} has minimum energy that is "
                                   f"more than 1kcal smaller than all the other states.\n Sorted energy for all states "
                                   f"are: "
                                   f"{sortedEnArray[:,1]}\n"
                                   f"Setting the state to:"
                                   f"isRotamer: {res2keep.isRotamer}\n")
                saveText = ''

                structure, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, changeVal, S, saveText)
                del S

   
    return structure,changeVal,skipVal,skipResInfo

def resolve_residue_ambiguities_in_one_structure(structure, set_original_centroid=False, generated_files=None,
                                                 pdb_file_num=None, outprefix='out'):

    if generated_files is None:
        generated_files = []
    if pdb_file_num is None:
        pdb_file_num = [0] # make it a list so that it's mutable for recursive call

    num_unknown_res = len(stp.get_unknown_residue_list(structure))

    # only stop when number of unknown residues is 0
    loop_count = 0
    while(num_unknown_res > 0):
        if gc.log_file:
            print(f'resolve_residue_ambiguities_in_one_structure Iteration: {loop_count}')

        # iterate over all the unknown residues and set as many states as possible. If a change is made or a state is set then chVal increments by 1 else skpVal increments by 1
        # chVal = total number of unknown states set in this function call
        # skpVal = total number of degenerate states found (these are the "ambiguous" residues in the paper)
        # skpInfo = all the info about that degenerate state
        structNew, chVal, skpVal, skpInfo = iterate_list_of_unknown_residues_and_set_states(structure)
        loop_count += 1

        # update the number of unknown residue here for the loop
        unknown_res_list = stp.get_unknown_residue_list(structNew)
        num_unknown_res = len(unknown_res_list)

        if num_unknown_res == 0: 
            if gc.log_file:
                print(f"No more unknown present\nBreak out of the while loop")
            structure = structNew
            break

        if (chVal == 0):
            # If change value =0, we need to branch the first unknown residue and go from there
            if gc.log_file:
                print(f"No states set after this iteration\nStart branching")

            if skpVal == 0:
                if gc.log_file:
                    print(f"Did not have any degenerate states ready for branching, meaning there "
                                   f"are no known residues nearby the rest of the unknowns\nSetting the rest of "
                                   f"unknowns to known: {unknown_res_list}\n")
                # in this case, there is nothing knonwn to determine the unknown residue, just set them as known and
                # exit
                for unknown_res in unknown_res_list:
                    structure[unknown_res.parent.parent.id][unknown_res.parent.id][unknown_res.id].isKnown = 1

                # update the number of unknown residue here for the loop
                unknown_res_list = stp.get_unknown_residue_list(structure)
                num_unknown_res = len(unknown_res_list)
                break



            # start branching from the first "ambiguous" residue
            # 0 for the first ambiguous residue, -2 to access all degenerate states that are evaluated
            # these info are being appened by "evaluate_degenerate_cases"
            all_degen_struc_names = skpInfo[0][-2]
            res_to_be_branch = skpInfo[0][0]
            if gc.log_file:
                print(f"The residue to be branched out: {res_to_be_branch}")
                print(f"Number of degenerate states previously generated to be used for branch: {len(all_degen_struc_names)}")

            for count, struc_name in enumerate(all_degen_struc_names):

                modelID = res_to_be_branch.parent.parent.id
                chainID = res_to_be_branch.parent.id

                structBranch = skpInfo[0][-1][struc_name]
                structBranch[modelID][chainID][res_to_be_branch.id].isKnown = 1
                if res_to_be_branch.resname in ['ASP', 'GLU']:
                    sUn = res_to_be_branch.parent.parent.parent
                    unknownASPpair = get_ASP_GLU_pair(res_to_be_branch, sUn, gc.deltaD)
                    modelID_ASPpair = unknownASPpair.parent.parent.id
                    chainID_ASPpair = unknownASPpair.parent.id
                    structBranch[modelID_ASPpair][chainID_ASPpair][unknownASPpair.id].isKnown = 1
                if gc.log_file:
                    print(f"Processing {res_to_be_branch}'s degenerate states: {count+1}/{len(all_degen_struc_names)}")
                    print("Start another 'resolve_residue_ambiguities_in_one_structure call'")
                _, num_unknown_res = resolve_residue_ambiguities_in_one_structure(structBranch,
                                                                                  set_original_centroid=set_original_centroid,
                                                                                  generated_files=generated_files,
                                                                                  pdb_file_num=pdb_file_num,
                                                                                  outprefix=outprefix)
            # it'd only get here when all branches are finished
            # we need to return here to avoid breaking out and write the final PDB again
            if num_unknown_res == 0:
                return generated_files, num_unknown_res


        else:
            # if chVal > 0 meaning at this iteration, there are residues that are set, keep looping
            structure = structNew
            del skpVal
            del skpInfo
            if gc.log_file:
                print(f"Number of states set after iteration {loop_count}: {chVal}\n")
                print(f"Current number of unknown residues: {num_unknown_res}\n")


    # when it get out of the while loop
    # meaning there are no more unknown residue, then write to PDB file
    fPDBname = f"{outprefix}_{pdb_file_num[0]}.pdb"
    fPDBfullPath = os.getcwd() + f"/{gc.out_folder}/{fPDBname}"

    if gc.log_file:
        print(f"No more unknown for this structure, Writing result to: {fPDBname}")

    stp.detect_clash_within_structure(structure)
    stp.detect_clash_within_residue_for_all_residues(structure)

    generated_files.append(fPDBfullPath)
    # keep a track on number of pdb files written
    pdb_file_num[0] += 1
    stp.write_to_PDB(structure, fPDBfullPath, removeHLP=True, set_original_centroid=set_original_centroid)
    return generated_files, num_unknown_res

