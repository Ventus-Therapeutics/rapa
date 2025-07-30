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
import Bio
import numpy as np

import copy
from collections import Counter
from Bio.PDB import *

import setup_protein as stp
import my_residue_atom as mra
import my_constants as mc
import hydrogen_placement_sp2 as hsp2
import hydrogen_placement_sp3 as hsp3
import close_atoms as cats


def setup_HIS(HISres, structure, HIStype = None, log_file=0, debug =0):

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
    lastSerial, hCoord = HIS_dict[HIStype](dict_struct[structName][modelID][chainID][resID], lastSerial, log_file=log_file, debug=debug)

    LP_dict = { 'HIE': [mc.hvysForLPsHIE, mc.LPSCnamesHIE], 'HID': [mc.hvysForLPsHID, mc.LPSCnamesHID]}
    try:
        hvys = LP_dict[HIStype][0]
        LPnameAll = LP_dict[HIStype][1]

        hsp2.place_lonepair(dict_struct[structName][modelID][chainID][resID], lastSerial, hvys, LPnameAll, debug=debug)
    except KeyError:
        if(log_file):stp.append_to_log(f"No Lone pairs to place for HIStype:{HIStype}, and unknown residue: {HISres} on chain: {HISres.parent}")
        if(debug):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"No Lone pairs to place for HIStype:{HIStype}, and unknown residue: {HISres} on chain: {HISres.parent}")
        pass

    
    return dict_struct[structName]


def setup_rotamer_in_structure(res, structure, placeHyd = True, placeLP = True, log_file=0, debug = 0):
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
            if(log_file): stp.append_to_log(f"Tried to remove hydrogen to setup a rotamer but no hydrogen were present for: {res} and chain: {res.parent}\n")
            if(debug):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Tried to remove hydrogen to setup a rotamer but no hydrogens were present for: {res} and chain: {res.parent}\n")
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
            lastSerial, hCoord = rotomerPlaceH[resInStruct.resname](resInStruct, lastSerial, log_file=log_file, debug=debug)
        except KeyError:
            if(log_file): stp.append_to_log(f"Tried to place hydrogens to set up a rotamer but no hydrogens for: {res} on chain: {res.parent} \n")
            if(debug):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Tried to place hydrogens to set up a rotamer but no hydrogens for: {res} on chain: {res.parent} \n", log_file=0, debug=0)
            pass

    LP_dict = {'ASP': [mc.hvysForLPsASP, mc.LPSCnamesASP], 'GLU': [mc.hvysForLPsGLU, mc.LPSCnamesGLU],
                'ASN': [mc.hvysForLPsASN, mc.LPSCnamesASN], 'GLN': [mc.hvysForLPsGLN, mc.LPSCnamesGLN],
                'HIE': [mc.hvysForLPsHIE, mc.LPSCnamesHIE], 'HID': [mc.hvysForLPsHID, mc.LPSCnamesHID]
    }


    if(placeLP):
        try:
            hvys = LP_dict[res.resname][0]
            LPnameAll = LP_dict[res.resname][1]
            hsp2.place_lonepair(res, lastSerial, hvys, LPnameAll, debug=debug)
        except KeyError:
            if(log_file): stp.append_to_log(f"Tried to place lone pairs but no Lone pairs for: {res} on chain: {res.parent} \n")
            if(debug):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f" Tried to place lone pairs but no Lone pairs for: {res} on chain: {res.parent} \n")
            pass

    resInStruct.isRotamer = 1

    return structure



def create_GLN_ASN_states(unRes, structure, log_file=0, debug=0):
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
    structDict["struct"+str(unRID)+unRName+"R"] = setup_rotamer_in_structure(res2Rotamer, structure2rotomer, log_file=log_file, debug=debug)

    if(debug):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"created all states for {unRes} of {unRes.parent} and added as a dictionary to: {structDict}")

    return structDict


def create_HIS_states(unRes, structure, log_file=0, debug=0):

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
        structDict[skey] = setup_HIS(unRes , structuresHIS[count], HIStype = name, log_file=log_file, debug=debug)
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
    structDict[skey] = setup_rotamer_in_structure( hie2Rotamer, structureHIErotomer, log_file=log_file, debug=debug)


    skey = "struct"+str(unRID)+"HIDR"
    skeyList.append(skey)
    structDict[skey] = setup_rotamer_in_structure( hid2Rotamer, structureHIDrotomer, log_file=log_file, debug=debug) 
    skey = "struct"+str(unRID)+"HIPR"
    skeyList.append(skey)
    structDict[skey] = setup_rotamer_in_structure( hip2Rotamer, structureHIProtomer, log_file=log_file, debug=debug)

    if(debug):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"created all states for {unRes} of {unRes.parent} and added as a dictionary to: {structDict}")
    
    return structDict


def create_ASH(structForASH, ASP_A, ASP_B, LPs, log_file=0, debug=0):

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
    
    if(log_file):
        stp.append_to_log(f"Need to create ASH from ASP_A:{ASP_A} from {ASP_A.parent}, and {ASP_B} from from {ASP_B.parent} with minimum distance: {minDist}.\n \
        All distances from lone pair(LP3/LP4 or LP5/LP6) associated with OD of ASP_A to OD of ASP_B: {dist_btw} are: {list2check}. Dropping LP:{lpChange} as it is associated with minimum distance \n")

    if(debug):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Need to create ASH from ASP_A:{ASP_A} from {ASP_A.parent}, and {ASP_B} from {ASP_B.parent} with minimum distance: {minDist}.\n \
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


def create_GLH(structForGLH, GLU_A, GLU_B, LPs, log_file=0, debug=0):

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

    if(log_file): stp.append_to_log(f"Need to create GLH from GLU_A:{GLU_A} from {GLU_A.parent}, and {GLU_B} from {GLU_B.parent} with minimum distance: {minDist}.\n \
    All distances from lone pair(LP3/LP4 or LP5/LP6) associated with OE of GLU_A to OE of GLU_B: {dist_btw} are: {list2check}. Dropping LP:{lpChange} as it is associated with minimum distance \n")

    if(debug):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Need to create GLH from GLU_A:{GLU_A} from {GLU_A.parent}, and {GLU_B} from from {GLU_B.parent} with minimum distance: {minDist}.\n \
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


def create_ASH_states(ASP_A_orig, ASP_B_orig, structure, log_file=0 ,debug=0):

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
        structDict[sNames[i-1]] = create_ASH(structForASH, ASP_list[i-1], ASP_list[-i], LPnames[(i-1)%2], log_file=log_file, debug=debug)

    if(debug):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"For ASPs: ASP_A:{ASP_A} of {ASP_A.parent} and ASP_B: {ASP_B.parent}, the ASH structures are created and stored in a dictionary: {structDict}")
    return structDict


def create_GLH_states(GLU_A_orig, GLU_B_orig, structure, log_file=0 ,debug=0):


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
        structDict[sNames[i-1]] = create_GLH(structForGLH, GLU1[i-1], GLU1[-i], LPnames[(i-1)%2], log_file=log_file, debug=debug)


    if(debug):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"For ASPs: ASP_A:{ASP_A} of {ASP_A.parent} and ASP_B: {ASP_B.parent}, the ASH structures are created and stored in a dictionary: {structDict}")

    return structDict


def get_ASP_GLU_pair(currASP_GLU, structure, ASP_GLUdist, log_file=0, debug=0):

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

    if(log_file):
        fLogName = stp.get_log_file_name()

    if(debug):
        stp.start_debug_file(__name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debug_file_name()

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

    
    if(debug):
        
        with open(fDebugName, "a") as fd:
            fd.write(f"Atom found in h-bonding distance neighborhood of {oxygenName1} belonging to {currASP_GLU} of {currASP_GLU.parent} is : {potUnknownAtom1}, and\n \
     Atom found in h-bonding distance neighborhood of {oxygenName2} belonging to {currASP_GLU} of {currASP_GLU.parent} is : {potUnknownAtom2} \n")


            fd.write(f"\n For both oxygen atoms the currrent ASP/GLU oxygen atom, current ASP/GLU oxygens residue, current atom residue chain, uat(atom belonging to unknown ASP/GLU in hbonding dist), uat residue, uats residues chain, distance between oxygen atom of current ASP/GLU and uat are the following:\n \
                    {unknownASP_GLUatomInfo}\n\n ")
            fd.flush()
    
    if(log_file):
        with open(fLogName, "a") as fLog:
            fLog.write(f"Atom found in h-bonding distance neighborhood of {oxygenName1} belonging to {currASP_GLU} of {currASP_GLU.parent} is : {potUnknownAtom1},\n \
                    and  Atom found in h-bonding distance neighborhood of {oxygenName2} belonging to {currASP_GLU} of {currASP_GLU.parent} is : {potUnknownAtom2} \n")
            fLog.flush()
   
    unknownASP_GLUlist = []

    #In the list od unknownASP atom that was just created-find its parent and append it
    for atom in unknownASP_GLUatom:
        unknownASP_GLUlist.append(atom.parent)
    
    #create a set so we can have the unique ASP residues
    unknownASP_GLUset = set(unknownASP_GLUlist)
    #since this code is meant for only diad!

    if(len(unknownASP_GLUset)> mc.numASPsGLUs-1):
        if(log_file):
            with open(fLogName, "a") as fLog:
                fLog.write(f"number of unique ASPs: {len(unknownASP_GLUset)}. Please check\n\n")
                fLog.flush()
       
        if(debug):
            with open(fDebugName, "a") as fd:
                fd.write(f"number of unique ASPs: {len(unknownASP_GLUset)}. Please check\n\n")
                fd.flush()

    
    
    if(not unknownASP_GLUset):
        unknownASP_GLUpair = None
    else: 
        unknownASP_GLUpair = list(unknownASP_GLUset)[0]
        
        for unknownASP_GLU in unknownASP_GLUset:
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"The unknownASP/GLU pair for {currASP_GLU} of {currASP_GLU.parent} is:  {unknownASP_GLU} of chain: {unknownASP_GLU.parent}\n")
                    fLog.flush()
            if(debug):
                with open(fDebugName, "a") as fd:
                    fd.write(f"The unknownASP/GLU pair for {currASP_GLU} of {currASP_GLU.parent} is:  {unknownASP_GLU} of chain: {unknownASP_GLU.parent}\n")
                    fd.flush()

    if(debug):
        stp.end_debug_file(__name__,sys._getframe().f_code.co_name)

    return unknownASP_GLUpair


def energy_of_donor_for_all_close_atoms(hvyAt, allCloseAtoms, chV_level, log_file=0, debug=0):
    """
    Objective: To compute energy interaction with neighboring atoms when my reference
                atom(current atom part of the unknown res) is a donor.
        I/P: hvyAt: heavy atom which is the part of unknown residue, allCloseAtoms: All 
                   close atoms we are considering
             con: name of the state, for example: GLN or GLNR or ASH_OD1A
             chV_level: to track and debug 
        O/P: enValList: Energy value list, and enSumTotal:  current total sum of the energy
 
    """

    if(log_file):
        fLogName = stp.get_log_file_name()

    if(debug):
        stp.start_debug_file( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debug_file_name()

        myHvyAt = mra.my_atom(hvyAt)
        myHvyAtBehav = myHvyAt.get_behavior().abbrev
        with open(fDebugName, "a") as fd:
            fd.write(f"behavior: do=donor, ac=acceptor,bo=both-donor and acceptor, tbd=to be decided\n")
            fd.write(f"The current heavy atom is: {hvyAt} (with behavior:{myHvyAtBehav}) of residue: {hvyAt.parent} of chain: {hvyAt.parent.parent}\n")
            fd.flush()

            for i in range(1, np.shape(allCloseAtoms)[0]):
                cAt= allCloseAtoms[i][0]
                cAtBehav = mra.my_atom(cAt).get_behavior().abbrev
                fd.write(f"closeAtom: {cAt}-behavior:{cAtBehav}, closeAtom residue:{cAt.parent}, close atom chain:{cAt.parent.parent} \n")
                fd.flush()
            fd.write(f"energy value details in the folder: energyInfo_ for each level and structure present\n")
            fd.flush()
            stp.end_debug_file(__name__,sys._getframe().f_code.co_name) 
    
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
        closeAtomBehav = mra.my_atom(currCloseAtom).get_behavior().abbrev
        currCloseResID = currCloseAtom.parent.id
        modelIDcloseAt = currCloseAtom.parent.parent.parent.id
        chainIDcloseAt = currCloseAtom.parent.parent.id
        currCloseRes = struct[modelIDcloseAt][chainIDcloseAt][currCloseResID] 

        if(currCloseRes.isSCHknown == 0):
            #This is so that the hvy atom parent is assumed to be known while computing SER close atoms. This flag is turned to 0 and after the computation
            hvyAt.parent.isKnown = 1
            lastSerial = list(struct.get_residues())[-1].child_list[-1].serial_number

            if(currCloseRes.resname=='SER' or currCloseRes.resname=='THR'):
                lastSerial, hLPCoords = hsp3.place_hydrogens_lonepairs_SER_THR(currCloseAtom.parent, lastSerial, log_file=log_file, debug=debug) 
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown = 0
                else:
                    currCloseRes.isSCHknown = 1

                hvyAt.parent.isKnown = 0

            elif(currCloseRes.resname in ['LYS', 'LYN']):
                lastSerial, hLPCoords = hsp3.place_hydrogens_lonepairs_LYS(currCloseAtom.parent, lastSerial, log_file=log_file, debug=debug)
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown = 0
                else:
                    currCloseRes.isSCHknown = 1
                hvyAt.parent.isKnown = 0
            elif(currCloseRes.resname == 'TYR'):
                lastSerial, hLPCoords = hsp2.place_hydrogens_TYR(currCloseAtom.parent, lastSerial, log_file=log_file, debug=debug)
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown = 0
                else:
                    currCloseRes.isSCHknown = 1
                hvyAt.parent.isKnown = 0
            else:
                continue
        #After attempting to place hydrogen/LP-if there is no known closeAtoms-then pls continue and try again in another iteration if it comes up!
        if(currCloseRes.isSCHknown==0): 
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"Current Close atom:{currCloseRes} is still unknown! Going on to the next close atom\n")
                    fLog.flush()
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
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"No backbone hydrogen is present for energy computations. Heavy atom: {hvyAt} belongs to {hvyAt.parent} of its chain: {hvyAt.parent.parent}. Perhaps its the first residue on the chain. \n")
                    fLog.flush()
        else:
            atomHs = cats.get_hydrogen_connected_to_donor(hvyAt)
            #going over all the hydrogen atoms connected to the heavy atom
            for hat in atomHs:
                if(closeAtomBehav=='do'):
                    enVal, enSum=cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive=0, atype='SP2', hName=hat.id, chV_levelVal=chV_level, log_file=log_file, debug=debug,pre_cal_donor_info=pre_cal_donor_info['hvyAt'])
                    enSumTotal = enSumTotal+enSum
                    enValList.append(enVal)

                elif(closeAtomBehav=='ac'):
                    #Next if it is an acceptor
                    enVal, enSum = cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive=1, atype='SP2',hName=hat.id, chV_levelVal=chV_level, log_file=log_file, debug=debug,pre_cal_donor_info=pre_cal_donor_info['hvyAt'])
                    enSumTotal = enSumTotal+enSum
                    enValList.append(enVal)

                elif(closeAtomBehav=='bo'):
                    #If both then-First acceptor
                    enVal0, enSum0 = cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive = 1, atype = 'SP2', hName = hat.id, chV_levelVal = chV_level, log_file=log_file ,debug=debug,  pre_cal_donor_info=pre_cal_donor_info['hvyAt'])
                    #Next donor
                    enVal1, enSum1 = cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive=0, atype='SP2', hName=hat.id, chV_levelVal=chV_level, log_file=log_file, debug=debug, pre_cal_donor_info=pre_cal_donor_info['hvyAt'])

                    enSumTotal = enSumTotal+enSum0+enSum1
                    enValList.append([enVal0, enVal1])
                    
                else:
                    continue
    return enValList,enSumTotal



def energy_of_acceptor_for_all_close_atoms(hvyAt, allCloseAtoms, chV_level, log_file=0, debug=0):

    """
    Objective: To compute energy interaction with neighboring atoms when my reference
               atom(current atom part of the unknown res) is an acceptor.
        I/P: hvyAt: heavy atom which is the part of unknown residue, allCloseAtoms: All
             close atoms we are considering
            chV_level: to track and debug, con: name of the residue state -to track and debug
        O/P: enValList: Energy value list, and enSumTotal:  current total sum of the energy
 
    """

    enSumTotal=0
    enValList=[]
    myHvyAt=mra.my_atom(hvyAt)
    LPAtoms=myHvyAt.get_lonepairs_atoms()

    struct=hvyAt.parent.parent.parent.parent

    if(debug):
        stp.start_debug_file( __name__, sys._getframe().f_code.co_name)
        fDebugName=stp.get_debug_file_name()
        myHvyAt = mra.my_atom(hvyAt)
        myHvyAtBehav = myHvyAt.get_behavior().abbrev
        with open(fDebugName, "a") as fd:
            fd.write(f"behavior: do=donor, ac=acceptor,bo=both-donor and acceptor, tbd=to be decided\n")
            fd.write(f"The current heavy atom is: {hvyAt} (with behavior: {myHvyAtBehav}) of residue: {hvyAt.parent} of chain: {hvyAt.parent.parent}\n")
            fd.flush()
            for i in range(1, np.shape(allCloseAtoms)[0]):
                cAt = allCloseAtoms[i][0]
                cAtBehav = mra.my_atom(cAt).get_behavior().abbrev
                fd.write(f"closeAtom: {cAt}-behavior:{cAtBehav}, closeAtom residue:{cAt.parent}, close atom chain:{cAt.parent.parent} \n")
                fd.flush()
            fd.write(f"energy value details in the folder: energyInfo_ for each level and structure present\n")
            fd.flush()
            stp.end_debug_file(__name__,sys._getframe().f_code.co_name) 

    pre_cal_acceptor_info = {'hvyAt': cats.get_info_for_acceptorAt_donorAt(hvyAt)}

    for i in range(1, np.shape(allCloseAtoms)[0]):
        currCloseAtom = allCloseAtoms[i][0]
        closeAtomBehav = mra.my_atom(currCloseAtom).get_behavior().abbrev

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
                lastSerial, hLPCoords=hsp3.place_hydrogens_lonepairs_SER_THR(currCloseAtom.parent, lastSerial, log_file=log_file, debug=debug)
                hvyAt.parent.isKnown=0
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown=0
                else:
                    currCloseRes.isSCHknown=1

            elif(currCloseRes.resname in ['LYS', 'LYN']):
                lastSerial, hLPCoords=hsp3.place_hydrogens_lonepairs_LYS(currCloseAtom.parent, lastSerial, log_file=log_file, debug=debug)
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown=0
                else:
                    currCloseRes.isSCHknown=1
                hvyAt.parent.isKnown=0

            elif(currCloseRes.resname=='TYR'):
                lastSerial, hLPCoords=hsp2.place_hydrogens_TYR(currCloseAtom.parent, lastSerial, log_file=log_file, debug=debug)
                if np.size(hLPCoords) == 0:
                    currCloseRes.isSCHknown=0
                else:
                    currCloseRes.isSCHknown=1
                hvyAt.parent.isKnown=0

            else:
                continue
    
        #After attempting to place hydrogen/LP-if there is no known closeAtoms-then pls continue and try again in another iteration if it comes up!
        if(currCloseRes.isSCHknown==0): 
            if(log_file): stp.append_to_log(f"Current Close atom:{currCloseRes} is still unknown! Going on to the next close atom \n")
            continue
        hvyAt.parent.isKnown=0
        #Going over all the lone pair atoms for acceptors. All acceptors have lone pairs in-order to accept a H atom
        for j in range(len(LPAtoms)):
                lp_vec = LPAtoms[j].get_vector()
                if(closeAtomBehav == 'do'):
                    #If close atom is a donor
                    enVal,enSum = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 1, atype = 'SP2', chV_levelVal = chV_level, log_file=log_file, debug=debug, pre_cal_acceptor_info=pre_cal_acceptor_info['hvyAt'])
                    #summing the total energy for a given heavy atom
                    enSumTotal = enSumTotal+enSum
                    enValList.append(enVal)                            
                elif(closeAtomBehav == 'ac'):
                    #If close atom is an acceptor
                    enVal, enSum = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 0, atype = 'SP2', chV_levelVal = chV_level, log_file=log_file, debug=debug, pre_cal_acceptor_info=pre_cal_acceptor_info['hvyAt'])
                    #summing the total energy for a given heavy atom
                    enSumTotal=enSumTotal+enSum
                    enValList.append(enVal)

                elif(closeAtomBehav == 'bo'):
                    #If close atom is both then taking the donor first(attractive=1) and then acceptor(attractive=0)
                    enVal0,enSum0 = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 1, atype = 'SP2', chV_levelVal = chV_level, log_file=log_file, debug=debug, pre_cal_acceptor_info=pre_cal_acceptor_info['hvyAt'])
                    enVal1,enSum1 = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 0, atype = 'SP2', chV_levelVal = chV_level, log_file=log_file, debug=debug, pre_cal_acceptor_info=pre_cal_acceptor_info['hvyAt'])

                    enSumTotal=enSumTotal+enSum0+enSum1
                    enValList.append([enVal0, enVal1])
                else:   
                    continue
                

    return enValList,enSumTotal



def compute_energy_for_given_atoms(resState, givenAtoms, chV_level, log_file=0, debug=0):

    """
    Objective: Given a configuration, compute energy sum of all the donors and acceptors present in the residue.
    I/P: resState: It is the unknown residue under consideration,
        givenAtoms: atoms we need to consider
        chV_level: to track and debug, con: name of the residue state -to track and debug

    O/P: enSumTotal: Total Energy sum of the configuration, 
         enSumForHvys: list of each heavy atom along with its energy sum
    """

    if(log_file):
        fLogName = stp.get_log_file_name()

    struct = resState.parent.parent.parent

    enSumTotal = 0
    enSumForHvys = []

   
    #Iterate over the given atoms in the input (as we need the energy for all those atoms with respect to its close atoms)
    for hvyAt in givenAtoms:
        myHvyAt = mra.my_atom(hvyAt)
        myHvyAtBehav = myHvyAt.get_behavior().abbrev

        #create a custom list to find all close atoms
        customList = stp.get_known_donor_acceptor_list_for_one_atom(struct, hvyAt, aaType = 'DONOR_ACCEPTOR_BOTH')
        ##Get a list of all close Atoms (which are donor, acceptor, and both) for the given hvy atom
        allCloseAtoms = cats.get_all_close_atom_info_for_one_atom(hvyAt, customList, dist_cutoff=mc.hbond_d)
        
        if(log_file):
            with open(fLogName, "a") as fLog:
                fLog.write(f"Evaluate given residue state, hvyAt: {hvyAt} and All close atoms: {allCloseAtoms}\n")
                fLog.flush()

        if(myHvyAtBehav == 'do'):
            #If the heavy atom is a donor the compute energy for all close atoms
            enValList, enSumTotalDonor = energy_of_donor_for_all_close_atoms(hvyAt, allCloseAtoms, chV_level, debug=debug)
            enSumTotal = enSumTotal + enSumTotalDonor
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"energy sum donor from func:{enSumTotalDonor} and current enSumTotal:{enSumTotal} \n")
                    fLog.write(str(enValList)+'\n')
                    fLog.flush()
        elif(myHvyAtBehav == 'ac'):
            #If heavyAtom is acceptor-compute energy for all its close atom.
            enValList, enSumTotalAcceptor = energy_of_acceptor_for_all_close_atoms(hvyAt, allCloseAtoms, chV_level, log_file=log_file, debug=debug)
            enSumTotal = enSumTotal + enSumTotalAcceptor
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"energy sum acceptor from func:{enSumTotalAcceptor} and current enSumTotal"
                               f":{enSumTotal}\n")
                    fLog.write(str(enValList)+'\n')
                    fLog.flush()
                
        else:
            continue

        enSumForHvys.append([hvyAt, enSumTotal])


    if(log_file):
        with open(fLogName, "a") as fLog:
            fLog.write(f"For this state:{resState} tot energy is: {enSumTotal}\n")
            fLog.write("************************************************************************\n")
            fLog.write("************************************************************************\n")
            fLog.flush()    

    return enSumTotal, enSumForHvys


def compute_energy_for_given_state(resState, chV_level, log_file=0, debug=0):

    """
    Objective: Given a configuration, compute energy sum of all the donors and acceptors present in the residue.
    I/P: config: It is the unknown residue under consideration.
    O/P: enSumTotal: Total Energy sum of the configuration, 
         enSumForHvys: list of each heavy atom along with its energy sum
    """     


    #Find list side chain active atoms, i.e donor/acceptor
    LOAAresState = mra.my_residue(resState).get_unknown_residue_acceptor_donor_atoms() 

    #Find the energy for all the side chain active atoms of the unknown residue
    enSumTotal, enSumForHvys = compute_energy_for_given_atoms(resState, LOAAresState, chV_level, log_file=log_file, debug=debug)

    if(debug):
        stp.start_debug_file( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debug_file_name()
        with open(fDebugName, "a") as fd:
            fd.write(f"For this state:{resState} (of {resState.parent}) where isRotamer: {resState.isRotamer}, given atoms: {LOAAresState} total energy is: {enSumTotal}, while the individual energies assembled as: heavy atom and energy value is: {enSumForHvys}\n")
            fd.write("************************************************************************\n")
            fd.write("************************************************************************\n")
            fd.flush() 
            stp.end_debug_file(__name__,sys._getframe().f_code.co_name) 

    return enSumTotal, enSumForHvys


def compute_energy_for_all_states(unknownRes, structStates, chV_level, log_file=0, debug=0):

    """
        objective: compute energy for all states of the given unknown residue 
        I/P:
        unknownRes: the unknown residue
        structStates: all the states of unknown residues
        O/P: energy list of all residue stats, and states
    """
    if(log_file):
        fLogName = stp.get_log_file_name()

    resStates = []
    energyList = []
    #because the modelID and chainID for a given residue-and its different states remain the same (They will be in same structure/chain/model)
    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id

    if(unknownRes.resname == 'ASP' or unknownRes.resname == 'GLU'):
        ##Get the 2 unknownresidue
        sUn = unknownRes.parent.parent.parent
        unknownASPpair = get_ASP_GLU_pair(unknownRes, sUn, mc.deltaD, log_file=log_file, debug=debug)
        unknownRes0 = [unknownRes, unknownRes, unknownASPpair, unknownASPpair]
    else:
        unknownRes0 = [unknownRes, unknownRes, unknownRes, unknownRes]

    #Going over all states of the unknown residue
    for count,con in enumerate(mc.unResDict[unknownRes.resname][0]):
        #accesing the struct state:
        structState = structStates['struct'+str(unknownRes0[count].id[1])+con]
        #accesing the residue state:
        resState = structState[modelID][chainID][unknownRes0[count].id]
        if(log_file):
            with open(fLogName, "a") as fLog:
                fLog.write("************************************************************************\n")
                fLog.write(f"State: {con}, with ID {unknownRes0[count].id[1]} and chain: {unknownRes0[count].parent} and isRotamer: {resState.isRotamer} \n")
                fLog.flush()
        if(debug):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f" ************************************************************************\n State: {con}, with ID {unknownRes0[count].id[1]} and chain: {unknownRes0[count].parent} and isRotamer: {resState.isRotamer} \n")
        resStates.append(resState)
        #compute energy for the given state:
        enSumTotalResState, enSumForHvys = compute_energy_for_given_state(resState, chV_level, log_file=log_file, debug=debug)
        energyList.append([resState, enSumTotalResState, structState, resState, con ])

    return energyList, resStates


def branch_structure(unknownRes, structure, log_file=0, debug=0):

    """
    objective: to create branch of the structure provided. This function creates all the possible options
               Note: we do not convert it into known at this point as it is only done when setting a state. Here we just create branch to analyze the different rotameric/protonation states.
    Input: -unknownRes-the unknown residue you want to branch
           -structure it belongs to
    Output: S - a dictionary of the multiple structures

    """

    #depending on the name of unknownRes create the structures
    if(unknownRes.resname=='HIS'):      
        S = create_HIS_states(unknownRes, structure, log_file=log_file, debug=debug)
    elif(unknownRes.resname=='ASN' or unknownRes.resname == 'GLN'):  
        S = create_GLN_ASN_states(unknownRes,structure, log_file=log_file, debug=debug)
    elif(unknownRes.resname=='ASP' or unknownRes.resname == 'GLU'):
        resPair = get_ASP_GLU_pair(unknownRes, structure, mc.deltaD, log_file=log_file, debug=debug)
        if(unknownRes.resname=='ASP'):
            S = create_ASH_states(unknownRes, resPair, structure, log_file=log_file, debug=debug)
        else:
            S = create_GLH_states(unknownRes, resPair, structure, log_file=log_file, debug=debug)
    else:
        S = None
    
    if(debug):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"For the unknown residue: {unknownRes} of {unknownRes.parent}, the following structures are created and stored in a dictionary: {S}")
    return S



def set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, branchedS, MSG, log_file=0, debug=0):

    """
    objective: To set the state in the given structure by converting it to known and removing it from the list of unknowns
        
    Input: -unknownRes: the unknown residue of concern
           -res2keep: the residue state you want to keep
           -nameOfS2keep: name of the structure you want to keep
           -unknownResMod: List of unknownRes modified-will remove the unknown residue from it as it is now known
           -changeVal: number of time changes occur
           -collectData: appending info about the new known to the data structure 
           -branchedS: the branched structure dict you want to use to assign the value to be known
           -MSG: The message you want to print on screen while setting the state
    
    Output: new and updated structure with known residue, updated list of unknown residue, changeVal and collect Data
    """
    struct = branchedS[nameOfS2keep]
    
    if(log_file):
        fLogName = stp.get_log_file_name()
        with open(fLogName, "a") as fLog:
            fLog.write(MSG+"\n")
            fLog.flush()

    changeVal+=1
    
    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id
    struct[modelID][chainID][unknownRes.id].isKnown = 1

    if(unknownRes.resname == 'ASP' or unknownRes.resname == 'GLU'):
        sUn = unknownRes.parent.parent.parent
        unknownASP_GLUpair = get_ASP_GLU_pair(unknownRes, sUn, mc.deltaD, log_file=log_file, debug=debug)
        modelIDpair = unknownASP_GLUpair.parent.parent.id
        chainIDpair = unknownASP_GLUpair.parent.id
        struct[modelIDpair][chainIDpair][unknownASP_GLUpair.id].isKnown = 1

    unknownResMod.remove(unknownRes)
    structure = copy.deepcopy(branchedS[nameOfS2keep])

    if(debug):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f" setting state for: {unknownRes} of {unknownRes.parent} \n \
                 residue kept:{res2keep} of {res2keep.parent}, isRotamer:{res2keep.isRotamer},\n \
                 name of structure kept:{nameOfS2keep},\n \
                 new list of unknowns: {unknownResMod},\n \
                 current change value:{changeVal},\n\
                 additional info inclued:[unknownRes, res2keep, unknownRes.parent.id, MSG ]:{collectData},\n \
                 branched structure is:{branchedS},\n \
                 updated structure kept is: {structure}\n\
                 message to print:{MSG}\n ")


    return structure, unknownResMod, changeVal


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


def get_degenerate_structure_names(degenArray, debug=0):

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
    
    if(debug):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"The degenerate structure names are: {degenStructNames}\n And more details are(resState, enSumTotalResState, structState, resState, con/name of the state ): {degenArray}\n")

    return degenStructNames


def get_HIP_energies(resHIP, resHIPR, chV_level, log_file=0, debug=0):

    """
    objective:-get energy for the HIP cases (original state + rotomer)
              -The two atoms: ND1 and NE2 needs to be treated separately as interaction energies
                              with both atoms require to be negetive for HIS to be a HIP case!

    Input: -resHIP: the original state of HIP
           -resHIPR: the rotamer state of HIP
           -ch_level: the changeValue + level value in a string format for debugging purpose

    Output: -enSumTotND1: energy for ND1 atom of HIP original state
            -enSumTotNE2: energy for NE2 atom of HIP original state
            -enSumTotND1Ro:  energy for ND1 atom of HIP rotomer state
            -enSumTotNE2Ro: energy for NE2 atom of HIP rotomer state

    """


    enSumTotND1, enSumForND1 = compute_energy_for_given_atoms(resHIP, [resHIP['ND1']], chV_level, log_file=log_file, debug=debug)
    enSumTotNE2, enSumForNE2 = compute_energy_for_given_atoms(resHIP, [resHIP['NE2']], chV_level, log_file=log_file, debug=debug)

    enSumTotND1Ro, enSumForND1Ro = compute_energy_for_given_atoms(resHIPR, [resHIPR['ND1']], chV_level, log_file=log_file, debug=debug)
    enSumTotNE2Ro, enSumForNE2Ro = compute_energy_for_given_atoms(resHIPR, [resHIPR['NE2']], chV_level, log_file=log_file, debug=debug)
    if(debug):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"computing energy value for HIPs: {resHIP} and {resHIPR} .The energies for the original state are the following.\n \
  For ND1, total sum and individual values are: {enSumTotND1}, {enSumForND1} \n \
  For NE2, total sum and individual values are: {enSumTotNE2}, {enSumForNE2} \n\n \
The energies for the rotamer state are the following.\n \
  For ND1 in rotamer state, total sum and individual values are: {enSumTotND1Ro}, {enSumForND1Ro} \n \
  For NE2 in rotamer state, total sum and individual values are: {enSumTotNE2Ro}, {enSumForNE2Ro} \n \
                ")

    return enSumTotND1, enSumTotNE2, enSumTotND1Ro, enSumTotNE2Ro



def evaluate_degenerate_cases( unknownRes,structure, S, energyArray, sortedEnArray, unknownResMod, changeVal, skipVal, skipResInfo, log_file=0, debug = 0):


    """
    objective: to update structures for the potential degenerate cases. By def: a degenerate case is
                where the two or more different states of a residue have energy difference less than 1 Kcal.
    
    Input: -unknownRes: the unknown residue with respect to which the degenerate case is developed
           -structure: the structure that needs to be updated.
           -S: it is a dictionary of multiple structures of the multiple degenerate cases.
           -LOCA_unknownRes: List of Close Atoms of the unknown residue
           -energyArray: energy  information for all the degenerate cases
           -sortedEnArray: energy information sorted in the min to max order
           -unknownResMod: the list of unknown residues that needs to be updated
           -changeVal: change value of the unknown->known residue needs to be tracked as degenerate
                        cases are fixed up
           -skipVal:skip value of the unknown->unknown residue(no change) needs to be tracked as well
           -collectData: collecting data for the changes being done
           -skipResInfo: collecting information where no changes are made-and the unknown residue is skipped


    Output:
           -structure: new and updated structure (If it is a truly degenerate case then structure remains the same as input structure)
           -unknownResMod: update list of unknown residue. If a residue state is fixed-it is then removed from this list.
           -changeVal: keeping track of changes that occur(unknown->known residues)
           -skipVal: keep track  of residues that are skipped in the list of unknowns
           -collectData: updated collectData of changes/no changes that occur
           -skipResInfo: updated info if the unknown residue's state is not yet set.This includes: unknown redisue, degenerate info, degenerate structure names, structure associated.
    """

    if(log_file):
        fLogName = stp.get_log_file_name()

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
            struct = copy.deepcopy(structure)
            sUn = unknownRes.parent.parent.parent
            unknownASP_GLUpair = get_ASP_GLU_pair(unknownRes, sUn, mc.deltaD, log_file=log_file, debug=debug)
            modelID_pair = unknownASP_GLUpair.parent.parent.id
            chainID_pair = unknownASP_GLUpair.parent.id


            struct[modelID][chainID][unknownRes.id].isKnown=1
            struct[modelID_pair][chainID_pair][unknownASP_GLUpair.id].isKnown=1
            res2keep=unknownRes 
            MSG=f'{unknownRes} and pair: {unknownASP_GLUpair} All energies = 0  '
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(MSG+"\n")
                    fLog.flush()

            unknownResMod.remove(unknownRes)
            unknownResMod.remove(unknownASP_GLUpair)
            structure = copy.deepcopy(struct)  
            changeVal +=1
            return structure, unknownResMod, changeVal, skipVal, skipResInfo

        nameOfS2keep = 'struct'+str(unknownRes.id[1])+nameOfRes2keep
        res2keep= S[nameOfS2keep][modelID][chainID][unknownRes.id]
        
        #min energy stored is the energy value picked depending on the residue name to keep
        ind = np.where(sortedResNamesArr[:]== res2keep.resname)[0][0]
        enValPick =sortedEnArray[ind][1]
       
        #string that will be printed out to the user along with other default info.
        condStr = "ALL ENERGIES = 0"

        msg2Usr = f"{condStr}, Saving residue: {res2keep} to residue state corresponding to minimum energy  i.e {enValPick} and isRotamer:{res2keep.isRotamer}"
        ###Setting the state-change value and skip values are updated in the function (set_state)
        structure, unknownResMod, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, S, msg2Usr, log_file=log_file, debug=debug)

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
         structure, unknownResMod, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, S, msg2Usr, log_file=log_file, debug=debug)

         del S

    elif(smallestEnergyNeg):
        #a degenerate case if the second smallest is not zero or >0
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
            structure, unknownResMod, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod,changeVal, S, msg2Usr, log_file=log_file, debug=debug)

            del S
        else:
            #A degenerate: as smallest and the second-smallest energies are zero
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
            indDegen = np.where(EnDiffWithMin<mc.ECutOff)
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
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"Skipping: {structure[modelID][chainID][unknownRes.id[1]]} as lowest vals:{sortedEnArray[0,1]} and {sortedEnArray[1,1]}, degenergate Info: {degenInfo} with {EnDiffWithMin[indDegen]}<{mc.ECutOff}\n")

                    fLog.write("###########################################################################\n")
                    fLog.flush()
                
            if(debug):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f' \n Skipping: {structure[modelID][chainID][unknownRes.id[1]]} as lowest vals:{sortedEnArray[0,1]} and {sortedEnArray[1,1]}, degenergate Info: {degenInfo} with {EnDiffWithMin[indDegen]}<{mc.ECutOff} \n')


    elif(allEnergyPos or allEnergyNeg):
            ##Degenerate case
            MinEn = sortedEnArray[0,1]

            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"All Energy Values are positive OR negative: {sortedEnArray}\n")
                    fLog.flush()

            #find the difference with minimum energy
            EnDiffWithMin =abs(sortedEnArray[:,1]-MinEn)
            #get the index where difference of energy is less than 1 kcal
            indDegen = np.where(EnDiffWithMin<mc.ECutOff)
            #use the index where diff of energy is less than 1 kcal to get degenNames, using the sortedResNamesArr
            degenNames = sortedResNamesArr[indDegen]
            #create the degenArray using the index where diff of energy<1Kcal in the sortedEnArray
            degenArray = sortedEnArray[indDegen]

            skipVal+=1
            #getting the degenerate structure names
            degenStructNames = get_degenerate_structure_names(degenArray)
            #updating skipResInfo
            skipResInfo.append([unknownRes, degenArray,degenStructNames,S])

            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"Please Skip: {structure[modelID][chainID][unknownRes.id[1]]} as lowest vals:{sortedEnArray[0,1]} and {sortedEnArray[1,1]}, degenergate names: {degenNames} and values are: {degenArray} with {EnDiffWithMin}<{mc.ECutOff}\n")
                    fLog.write("###########################################################################\n")
                    fLog.flush()

            if(debug):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Please Skip: {structure[modelID][chainID][unknownRes.id[1]]} as lowest vals:{sortedEnArray[0,1]} and {sortedEnArray[1,1]}, degenergate names: {degenNames} and values are: {degenArray} with {EnDiffWithMin}<{mc.ECutOff}\n ###########################################################################\n")
                
    else:
           if(log_file): 
               with open(fLogName, "a") as fLog:
                   fLog.write("ERROR: SHOULD NOT GO THROUGH DEGENERATE CASES!\n")
                   fLog.flush()
           if(debug):
               stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"ERROR: SHOULD NOT GO THROUGH DEGENERATE CASES. \n")
           os.exit()
           
    
    return structure, unknownResMod,changeVal, skipVal, skipResInfo


def evaluate_HIP_cases(unknownRes, structure, S, unknownResMod, changeVal, skipVal, skipResInfo, chV_level, log_file=0, debug =0) :
    """
    objective: To investigate the HIS could be HIP/HIP rotamer
    I/P:-unknownRes: the unknown HIS to investigate
        -structure: structure to update
        -S: is a dictionary of possible structures corresponding to HIP/HIE/HID and rotamers
        -unknownResMod: list of unknown residues that needs to be updated
        -changeVal: change value of the unknown->known residue needs to be tracked as degenerate
                        cases are fixed up
        -skipVal:skip value of the unknown->unknown residue(no change) needs to be tracked as well
        -collectData: collecting data for the changes being done
        -skipResInfo: collecting information where no changes are made-and the unknown residue is skipped.This includes: unknown residue, degenerate info, degenerate structure names, structure associated
        -chV_level: a combination string for changeVal and level. This helps to monitor code progress.

    O/P:
        -structure: new and updated structure
        -unknownResMod: update list of unknown residue. If a residue state is fixed-it is then removed from this list.
        -changeVal: keeping track of changes that occur(unknown->known residues)
        -skipVal: keep track  of residues that are skipped in the list of unknowns
        -collectData: updated collectData of changes/no changes that occur
        -skipResInfo: updated info if the unknown residue's state is not yet set!
        -S:update the dictionary structure
        -HIPset: if HIS is set as HIP-> then set this flag to 1/true
        -HIPdegen: if HIS is a degenerate case between HIP and HIP rotamer then set this flag to true!!

    """    

    if(log_file):
        fLogName = stp.get_log_file_name()

    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id

    sHIP = 'struct'+str(unknownRes.id[1])+'HIP' 
    resHIP = S[sHIP][modelID][chainID][unknownRes.id]

    sHIPR = 'struct'+str(unknownRes.id[1])+'HIPR' 
    resHIPR = S[sHIPR][modelID][chainID][unknownRes.id]
    HIPset = 0
    HIPdegen = 0


    enSumTotND1, enSumTotNE2, enSumTotND1Ro, enSumTotNE2Ro   = get_HIP_energies(resHIP, resHIPR, chV_level, log_file=log_file, debug=debug)    
    sumOrigState = enSumTotND1 + enSumTotNE2
    
    sumRotamer = enSumTotND1Ro + enSumTotNE2Ro
    origStateHIP = enSumTotND1< -1*mc.ECutOff and enSumTotNE2< -1*mc.ECutOff
    rotamerStateHIP = enSumTotND1Ro< -1*mc.ECutOff and enSumTotNE2Ro<-1*mc.ECutOff

    #Checking the degenerate case
    if(origStateHIP and rotamerStateHIP and abs(sumOrigState - sumRotamer)<1*mc.ECutOff and (enSumTotND1 !=0 and enSumTotNE2 !=0 and enSumTotND1Ro !=0 and enSumTotNE2Ro !=0 ) ):
          if(log_file):  
              with open(fLogName, "a") as fLog:
                  fLog.write(f"HIP:{unknownRes} of {unknownRes.parent} is DEGENERATE!! \n")
                  fLog.flush()
          if(debug):
              stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"HIP:{unknownRes} of {unknownRes.parent} is DEGENERATE!! \n")
          #create the list and array of energy information
          energyList = [[resHIP, sumOrigState,S[sHIP], resHIP,'HIP'], [resHIPR, sumRotamer, S[sHIP],resHIPR ,'HIPR']]
          energyArray = np.array(energyList)
          sortedEnList = sorted(energyList,key=lambda x: (x[1]))
          sortedEnArray = np.array(sortedEnList)


          #check the degenerate cases
          skipValb4check = skipVal
          structure, unknownResMod,changeVal, skipVal, skipResInfo = evaluate_degenerate_cases( unknownRes,
                                                                                                structure, S, energyArray, sortedEnArray, unknownResMod, changeVal, skipVal, skipResInfo, log_file=log_file, debug=debug)
          if(abs(skipValb4check - skipVal)==0):
              HIPdegen =0
          else:
              HIPdegen = 1
              del S['struct'+str(unknownRes.id[1])+'HIE' ]
              del S['struct'+str(unknownRes.id[1])+'HIER' ]
              del S['struct'+str(unknownRes.id[1])+'HID' ]
              del S['struct'+str(unknownRes.id[1])+'HIDR' ]

          return structure, unknownResMod, changeVal, skipVal, skipResInfo, S, HIPset, HIPdegen

    elif(enSumTotND1< -1*mc.ECutOff and enSumTotNE2< -1*mc.ECutOff):
        ##check if the original state is a possibility
        ##Also check if the closest atom is OG from SER or OG1 from THR. If it is then HIP cannot be the state.
        ##Because OG/OG1 are poor acceptors
        if(log_file):
            with open(fLogName, "a") as fLog:
                fLog.write(f"possibility of original state HIP as the two energy values are: {enSumTotND1} and {enSumTotNE2}\n")
                fLog.flush()
        if(debug):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"possibility of original state HIP as the two energy values are: {enSumTotND1} and {enSumTotNE2}")
        myHIP = mra.my_residue(resHIP)
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
                if(log_file):
                    with open(fLogName, "a") as fLog:
                        fLog.write(f"cannotBeHIP: {cannotBeHIP}\n")
                        fLog.flush()

                if(debug):
                    stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"cannotBeHIP: {cannotBeHIP}\n")
                del S['struct'+str(unknownRes.id[1])+'HIP' ]
                del S['struct'+str(unknownRes.id[1])+'HIPR' ]

            else:
                res2keep = resHIP
                nameOfS2keep = sHIP
                saveText = f"Saving residue: {res2keep.resname} to residue state corresponding to minimum energy i.e ND1Ro: {enSumTotND1} + NE2Ro: {enSumTotNE2} isRotamer:{res2keep.isRotamer}"
                structure, unknownResMod, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, S, saveText, log_file=log_file, debug=debug)
                HIPset = 1 

                del S['struct'+str(unknownRes.id[1])+'HIE' ]
                del S['struct'+str(unknownRes.id[1])+'HIER' ]
                del S['struct'+str(unknownRes.id[1])+'HID' ]
                del S['struct'+str(unknownRes.id[1])+'HIDR' ]
                del S['struct'+str(unknownRes.id[1])+'HIPR' ]
        
        else: 
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"ERROR: Investigating HIP-where energies for both nitrogen's<-{1*mc.ECutOff} but close atoms = 0. PLS CHECK CODE \n")
                    fLog.flush()
            
            if(debug):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"ERROR: Investigating HIP-where energies for both nitrogen's<-1*{mc.ECutOff} but close atoms = 0. Check more details. \n")
            os.exit()

    elif(enSumTotND1Ro< -1*mc.ECutOff and enSumTotNE2Ro<-1*mc.ECutOff):
        ##check if the rotamer state is a possibility
        ##Check if the closest atom is OG from SER or OG1 from THR. If it is then HIP cannot be the state
        ##Because OG/OG1 are poor acceptors
        if(log_file):
            with open(fLogName, "a") as fLog:
                fLog.write("possibility of rotamer state HIP...\n")
                fLog.flush()
        if(debug):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"possibility of rotamer state HIP...\n")
        myHIPR = mra.my_residue(resHIPR)

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
                if(log_file):
                    with open(fLogName, "a") as fLog:
                        fLog.write(f"cannotBeHIPR: {cannotBeHIPR} \n")
                        fLog.flush()

                if(debug):
                    stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"cannotBeHIPR: {cannotBeHIPR} \n")
                del S['struct'+str(unknownRes.id[1])+'HIP' ]
                del S['struct'+str(unknownRes.id[1])+'HIPR' ]

            else: 
                #It is HIP rotamer:
                res2keep = resHIPR
                nameOfS2keep = sHIPR
                saveText = f"Saving residue: {res2keep.resname} to residue state corresponding to minimum energy i.e ND1Ro: {enSumTotND1Ro} + NE2Ro: {enSumTotNE2Ro} isRotamer:{res2keep.isRotamer}"
                structure, unknownResMod, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, S, saveText, log_file=log_file, debug=debug)
                HIPset = 1 

                del S['struct'+str(unknownRes.id[1])+'HIE' ]
                del S['struct'+str(unknownRes.id[1])+'HIER' ]
                del S['struct'+str(unknownRes.id[1])+'HID' ]
                del S['struct'+str(unknownRes.id[1])+'HIDR' ]
                del S['struct'+str(unknownRes.id[1])+'HIP' ]

        else: 
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"Investigating HIP Rotamer-where energies for both nitrogen's<-{1*mc.ECutOff} but close atoms = 0. Check more details. \n")
                    fLog.flush()
            if(debug):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Investigating HIP Rotamer-where energies for both nitrogens<-{1*mc.ECutOff} but close atoms = 0. Check more details. \n")
            os.exit()

    else:
        del S['struct'+str(unknownRes.id[1])+'HIP' ]
        del S['struct'+str(unknownRes.id[1])+'HIPR' ]

    return structure, unknownResMod, changeVal, skipVal, skipResInfo, S, HIPset, HIPdegen


def iterate_list_of_unknown_residues_and_set_states(structure, level, chV_level, numCount=1, log_file=0, debug=0):
    """
    objective: To iterate over a given structure and to set states depending on energy computations
    I/P: structure: structure of a protein
         level: the vertical level we are at
         chV_level: a string that combines: change value in a given structure and level value. Used for debug/tracking purposes.
         numCount: structure number on this level

    O/P: structure: updated structure
         changeVal: number of changes that occur
         skipVal: num of residues skipped
         skipResInfo: collecting information where no changes are made-and the unknown residue is skipped. This includes: unknown residue, degenerate info, degenerate structure names, structure associated
    """

    if(log_file):
        fLogName = stp.get_log_file_name()

    skipVal = 0
    changeVal =0 

    skipResInfo = []

    unknownResIter = stp.get_unknown_residue_list(structure)
    unknownResMod = copy.deepcopy(unknownResIter)

    lenUnResOrig = len(unknownResIter)

    unknownResNames = []
    for res in unknownResIter:
        unknownResNames.append(res.resname)

    uniqRes = Counter(unknownResNames).items()

    if(log_file):
        with open(fLogName, "a") as fLog:
            fLog.write(f"\n Level:{level}, structure number on this level: {numCount}, unknown residue to iterate over: {unknownResIter}, and its length: {len(unknownResIter)}\n\n")
            fLog.write(f"\n The unknowns present: {uniqRes} \n\n")
            fLog.flush()

    
    if(debug):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"\n Level:{level}, structure number on this level: {numCount}, unknown residue to iterate over: {unknownResIter}, and its length: {len(unknownResIter)}\n\n The unknowns present: {uniqRes} \n\n")
    
    
    for count, unknownResOrig in enumerate(unknownResIter):    
        modelID = unknownResOrig.parent.parent.id
        chainID = unknownResOrig.parent.id
        unknownRes = structure[modelID][chainID][unknownResOrig.id]
        if(unknownRes.isKnown == 1):continue
        
        if(log_file):
            with open(fLogName, "a") as fLog:
                fLog.write("\n###############################################################################\n")
                fLog.write("###############################################################################\n\n")
                fLog.write(f"unknown residue number: {count+1}/{lenUnResOrig} and unknown res is: {unknownRes} and its known val:{unknownRes.isKnown} and is rotamer:{unknownRes.isRotamer}, Skip value:{skipVal}, ChangeVal: {changeVal}\n")
                fLog.flush()
        
        if(debug):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f" ###############################################################################\n ###############################################################################\n\n Number: {count+1}/{lenUnResOrig} and unknown res is: {unknownRes} and its known val:{unknownRes.isKnown} and is rotamer:{unknownRes.isRotamer}, Skip value:{skipVal}, ChangeVal: {changeVal}\n")
    
        current_unknown_res = mra.my_residue(unknownRes)
        #Get list of active atoms, list of close atoms, and number of close atoms
        LOAA_unknownRes = current_unknown_res.get_unknown_residue_acceptor_donor_atoms()
        ##This adds close points for rotamer HIS as well:
        if(unknownRes.resname == 'HIS'):
            LOAA_unknownRes.append(structure[modelID][chainID][unknownRes.id]['CD2'])
            LOAA_unknownRes.append(structure[modelID][chainID][unknownRes.id]['CE1'])
    
        #Get all the close atoms for the atoms in concern
        #This list of close atoms may not contain only known atoms
        LOCA_unknownRes, dim = cats.get_list_of_close_atoms_for_list_of_atoms(LOAA_unknownRes,
                                                                               'DONOR_ACCEPTOR_BOTH_TBD',
                                                                              include_self=False)
        # For HIS, ASN, GLN, if there are no polar atoms (unknown residues included), if it's HIS, set HIS-> HIE,
        # if ASN/GLN keep them as they are and mark them known
        if(all(x == 0 for x in dim)):
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"No Close Atoms Found!\n\n")
                    fLog.flush()
            if(debug):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"No Close Atoms Found!\n\n")
            
            if unknownRes.resname=='HIS':
                setup_HIS(unknownRes, structure, 'HIE', log_file=log_file, debug=debug)

            structure[modelID][chainID][unknownRes.id].isKnown = 1 
            unknownRes.isKnown = 1
            changeVal +=1
            unknownResMod.remove(unknownRes)
        else:
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write(f"checked for close atoms and close atoms are present!!\n\n")
                    fLog.flush()
            if(debug):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"checked for close atoms and close atoms are present!!\n\n")
            #create branch of a given unknown residue.
            S = branch_structure(unknownRes, structure, log_file=log_file, debug=debug)

            energyList, resStates = compute_energy_for_all_states(unknownRes, S, chV_level, log_file=log_file, debug=debug)

            energyArray = np.array(energyList, dtype=object)
            sortedEnList = sorted(energyList,key=lambda x: (x[1]))
            sortedEnArray = np.array(sortedEnList, dtype=object)

            allEnergyZero = all(c == 0 for c in energyArray[:,1])    ##All close atoms have zero energy

            if(unknownRes.resname == 'HIS' and not(allEnergyZero)): 
                #If HIS and all energy values are not zero-test the possibility of HIP:
                structure, unknownResMod, changeVal, skipVal, skipResInfo, S, HIPset, HIPdegen = evaluate_HIP_cases(
                    unknownRes, structure, S,  unknownResMod, changeVal, skipVal, skipResInfo, chV_level, log_file=log_file, debug=debug)

                if(HIPdegen==1 or HIPset==1):
                    if(log_file):
                        with open(fLogName, "a") as fLog:
                            fLog.write("Continuing to the next unknown in the for loop as HIP is set or HIP is degenerate\n")
                            fLog.flush()

                    if(debug):
                        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Continuing to the next unknown in the for loop as HIP is set or HIP is degenerate\n")
                    continue

                if(log_file):
                    with open(fLogName, "a") as fLog:
                        fLog.write("continuing as HIP is not an option!!\n") ##continue in the for loop
                        fLog.flush()

                if(debug):
                    stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"continuing as HIP is not an option!!\n")

            else:
                #If not a HIS/all HIS energy values are zero. Declare HIP is not possible.
                if(log_file):
                    with open(fLogName, "a") as fLog:
                        fLog.write(f"Not checking for HIP- as I am {unknownRes} of {unknownRes.parent} and currently am not on HIS or all the HIS energies are zero\n\n")
                        fLog.flush()
                if(debug):
                    stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Not checking for HIP- as I am {unknownRes} of {unknownRes.parent} and currently am not on HIS or all the HIS energies are zero\n\n")
    #########Take the smallest value and subtract from all other values. Find out how many values are<1
            MinEn = sortedEnArray[0,1]
            #The first value is the minimum. So starting from second Val
            diffWithMin = abs(sortedEnArray[1:,1]-MinEn)

            ##checking if the difference between the smallest two energy value is less than 1-thus a possible degeneracy!
            degenFound = any(diffWithMin< mc.ECutOff) 
###################################IF DEGENERATE ###############################################################
            if(degenFound):
                #if smallest two energies are less than ECutOff then explore the possibility of degeneracy 
                structure, unknownResMod,changeVal, skipVal, skipResInfo = evaluate_degenerate_cases(unknownRes, structure, S, energyArray, sortedEnArray, unknownResMod, changeVal, skipVal, skipResInfo, log_file=log_file, debug=debug)
            else:
                ##If not degenerate, pick the state with the smallest energy value:
                res2keep = sortedEnArray[0,0] ##Picking the smallest
                ind = 0
                moreInfoName = sortedEnArray[0,-1]
                nameOfS2keep = get_structure_name(res2keep, moreInfo = moreInfoName)

                enValPick =sortedEnArray[ind][1]
                saveText = f"State Set! The energies of :{sortedEnArray[0,0]} is {sortedEnArray[0,1]} and {sortedEnArray[0,0]} is {sortedEnArray[1,1]}, Saving residue: {res2keep.resname} to residue state corresponding to minimum energy  i.e {enValPick} and isRotamer:{res2keep.isRotamer}"

                structure, unknownResMod, changeVal = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, S, saveText, log_file=log_file, debug=debug)
                del S

   
    return structure,changeVal,skipVal,skipResInfo


def resolve_residue_ambiguities_in_structure(structure,fOutName = 'out', maxLevel=5, chValMax=5,set_original_centroid=False, log_file=0, debug=0): 
    """
    objective: resolve ambiguity of the unknown residues in a given structure
    I/P: -structure: structure of the protein(it is a class data structure)
         -maxLevel: max number of levels we want to traverse
         -chValMax: Max number of change value in each level to accommodate

    O/P: -pdbFileNum: number of PDB files created
         -listOfStructsAll: list of all structs created during the traversal
         -skipInfoAll: Information of all residues skipped in each structure formed
         -skipValAll: skip values during each structure formation
    """
    

    listOfStructsCurr=[structure]
    listOfStructsAll=[structure]
    skipInfoAll=[]
    skipValAll=[]
    filesGen=[]

    pdbFileNum=0
    #pdbOutFolder = stp.get_pdb_out_folder()

    if(log_file):
        fLogName = stp.get_log_file_name()

    for level in range(maxLevel):
        level+=1
        if(log_file):
            with open(fLogName, "a") as fLog:
                fLog.write(f"Starting a new level:{level}\n")
                fLog.flush()
        
        if(debug):
            stp.append_to_debug( __name__, sys._getframe().f_code.co_name,f"Starting a new level:{level}\n")

        #current level S Len is the number of structures in a given level. Horizontal traversal first and then vertical
        #This is number of structures in horizontal traversal
        currLevelSLen = len(listOfStructsCurr)

        if(currLevelSLen == 0):
            #if there are no structures in the current level then break out as all the files to be written are also done.
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write("Current S length=0! No more structures to explore\n") 
                    fLog.flush()
            if(debug):
                stp.append_to_debug( __name__, sys._getframe().f_code.co_name,"Current structure list(S) length =0! No more structs to explore\n")

            break
        if(level > maxLevel-1): 
            if(log_file):
                with open(fLogName, "a") as fLog:
                    fLog.write("ERROR: Exceeding max level limit!\n")
                    fLog.flush()
            if(debug):
                stp.append_to_debug( __name__, sys._getframe().f_code.co_name,"ERROR LEVEL: Exceeding max level limit!\n")
        
        #to store the newly created list of structures for the next level
        LOS_newLevel = []
        LOS_newLevel_dicts = []
        if(log_file):
            with open(fLogName, "a") as fLog:
                fLog.write(f"In the OUTER most loop where level is: {level} and length of structure list(S):{currLevelSLen}\n")
                fLog.flush()
        if(debug):
            stp.append_to_debug( __name__, sys._getframe().f_code.co_name,f"In the outer most loop where level is: {level} and length of structure list(S):{currLevelSLen}\n")

        #Need to iterate over each structure in the current level:
        for st in range(currLevelSLen):
            chVal = 1
            count = 0 
            structCurr = listOfStructsCurr[st]
            
            #Going over max amount of change Values that can occur
            for chV in range(chValMax):
                chV+=1
                #if exceeding max-human intervention is required.
                if(chV > chValMax-1): 
                    if(log_file):
                        with open(fLogName, "a") as fLog:
                            fLog.write("\n##########################################\n")
                            fLog.write("\nERROR CHV: Exceeding max chV level limit!\n")
                            fLog.write("\n##########################################\n")
                            fLog.flush()

                    if(debug):
                        stp.append_to_debug( __name__, sys._getframe().f_code.co_name,\
                                        "\n##########################################\n \
                                         \nERROR CHV: Exceeding max chV level limit!\n \
                                         \n##########################################\n")



                    
                if(chVal>0):
                    #as long as change value is positive, keep iterating over list of unknown residues
                    count +=1
                    chV_level = "level_"+str(level)+"_chV_" + str(chV) +"_structureNum_"+str(st)

                    if(log_file):
                        with open(fLogName, "a") as fLog:
                            fLog.write("\n###########################################################\n #############################################################\n")
                            fLog.write(f"chV(in range of chValMax or max amount of change Values that can occur ={chValMax}): {chV}, chVal(or number of residue changed in this level):{chVal} current level: {level}, st:{st}/{currLevelSLen-1}, counting number of times a change occurs: {count-1}\n")
                            fLog.flush()

                    if(debug):
                        stp.append_to_debug( __name__, sys._getframe().f_code.co_name,\
f"\n###########################################################\n #############################################################\n chV(in range of chValMax or max amount of change Values that can occur ={chValMax}): {chV}, chVal(or number of residue changed in this level):{chVal} current level: {level}, st:{st}/{currLevelSLen-1}, counting number of times a change occurs: {count-1}\n")
 

                    #iterate over all the unknown residues and set as many states as possible. If a change is made or a state is set then chVal increments by 1 else skpVal increments by 1
                    structNew, chVal, skpVal, skpInfo = iterate_list_of_unknown_residues_and_set_states(structCurr,level, chV_level, numCount=count, log_file=log_file, debug=debug)
                    skipInfoAll.append(skpInfo)
                    skipValAll.append(skpVal)

                    if(log_file):
                        with open(fLogName, "a") as fLog:
                            fLog.write(f"chVal after goin through the iterative loop: {chVal}\n")
                            fLog.flush()

                    if(debug):
                        stp.append_to_debug( __name__, sys._getframe().f_code.co_name,\
                                f"chVal after goin through the iterative loop: {chVal}\n")


                    if(chVal == 0):
                        #If change value = 0, then compute list of unknown residues else update structCurr to structNew and repeat.
                        LOURch0 = stp.get_unknown_residue_list(structNew)
                        #If change value  =0, and there are still unknowns in the list-it is time to branch!!
                        if(LOURch0):
                           if(log_file):
                               with open(fLogName, "a") as fLog:
                                   fLog.write(f"level:{level},In resolve ambiguity, in List Of Unknown Residue change0 (LOURcho): unknown {LOURch0} and length: {len(LOURch0)}\n")
                                   fLog.flush()
                           if(debug):
                               stp.append_to_debug( __name__, sys._getframe().f_code.co_name,\
                                f"chVal after goin through the iterative loop: {chVal}\n")


                    if(chVal == 0):
                        #If change value=0, then compute list of unknown residues else update structCurr to struct new and repeat.
                        LOURch0 = stp.get_unknown_residue_list(structNew)
                        #If change value=0, and there are still unknowns in the list-it is time to branch.
                        if(LOURch0):
                           if(log_file):
                               with open(fLogName, "a") as fLog:
                                fLog.write(f"level:{level},In resolve ambiguity, in List Of Unknown Residue change0 (LOURcho): unknown {LOURch0} and length: {len(LOURch0)}\n")

                                fLog.flush()
                           if(debug):
                               stp.append_to_debug( __name__, sys._getframe().f_code.co_name,\
                                       f"level:{level},In resolve ambiguity, in List Of Unknown Residue change0 (LOURcho): unknown {LOURch0} and length: {len(LOURch0)}\n")
                           #Create the branch:
                           #Get the name of structure of for the degenerate cases 
                           namesOfStructs = skpInfo[0][-2]
                           S2branch = {}
                           for name in namesOfStructs:
                                modelID = LOURch0[0].parent.parent.id
                                chainID = LOURch0[0].parent.id

                                structBranch = skpInfo[0][-1][name]
                                structBranch[modelID][chainID][LOURch0[0].id].isKnown = 1
                                if(LOURch0[0].resname == 'ASP' or LOURch0[0].resname == 'GLU'):
                                    sUn = LOURch0[0].parent.parent.parent
                                    unknownASPpair = get_ASP_GLU_pair(LOURch0[0], sUn, mc.deltaD, log_file=log_file, debug=debug)
                                    modelID_ASPpair = unknownASPpair.parent.parent.id
                                    chainID_ASPpair = unknownASPpair.parent.id
                                    structBranch[modelID_ASPpair][chainID_ASPpair][unknownASPpair.id].isKnown=1
                                name2ref = name +'_level_' +str(level)
                                S2branch[name2ref] = structBranch

                           ##Make the first residue "known" as we are taking all possible states
                           #save as dictionary for easy access while inspection
                           LOS_newLevel_dicts.append(S2branch)
                           LOS_newLevel.append(list(S2branch.values()))
                           listOfStructsAll.append(S2branch)
                
                           if(log_file):
                               with open(fLogName, "a") as fLog:
                                   fLog.write(f"printing all list of structs: {listOfStructsAll}, its length: {len(listOfStructsAll)}\n")
                                   fLog.write(f"Breaking out of loop as ChVal = {chVal} and branch of S has been created: {S2branch}\n")
                                   fLog.flush()

                           if(debug):
                               stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"printing all list of structs: {listOfStructsAll}, its length: {len(listOfStructsAll)}\n Breaking out of loop as ChVal = {chVal} and branch of S has been created: {S2branch}\n")
                           break###Breaking out of chValMax for loop
                        else:
                            ###If there are no more unknown residue, pls WRITE2PDB
                            #fPDBfullPath = os.getcwd()+f"/{pdbOutFolder}/{fOutName}_{pdbFileNum}.pdb"
                            fout=stp.get_output_folder_name()
                            #import code
                            #code.interact(local=locals())
                            fPDBfullPath = os.getcwd()+f"/{fout}/{fOutName}_{pdbFileNum}.pdb"
                            fPDBname = f"{fOutName}_{pdbFileNum}.pdb"
                            
                            if(log_file):
                                with open(fLogName, "a") as fLog:
                                    fLog.write(f"Writing a file: {fPDBname} since currently my state is: chV: {chV}, chVal:{chVal} current level: {level}, st:{st}/{currLevelSLen-1}, counting number of times a change occurs: {count}\n")  
                                    fLog.flush()
                            
                            if(debug):
                                stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"I am now writing a file: {fPDBname} since currently my state is: chV: {chV}, chVal:{chVal} current level: {level}, st:{st}/{currLevelSLen-1}, counting number of times a change occurs: {count}\n")
        
                            stp.detect_clash_within_structure(structNew, log_file=log_file)
                            stp.detect_clash_within_residue_for_all_residues(structNew,  log_file=log_file)
        
                            filesGen.append(fPDBfullPath)
                            #keep a track on number of pdb files written
                            pdbFileNum+=1  
                            stp.write_to_PDB(structNew, fPDBfullPath, removeHLP=True,set_original_centroid=set_original_centroid, log_file=log_file, debug=debug) 
                            
                            break
                    else:

                        if(log_file):
                            #If change value is not yet zero-keep iterating over by updating the current struct to new struct   
                            with open(fLogName, "a") as fLog:
                                fLog.write("\nNow assigning the new structure to current structure\n")
                                fLog.flush() 

                        if(debug):
                            stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"\n Now assigning the new structure to current structure\n")
                        structCurr = structNew
                else:
                     if(log_file):
                         with open(fLogName, "a") as fLog:
                         #If change value is not greater than 0 then break out of the loop.
                             fLog.write(f"change value is not greater than 0. I am breaking out. \n")
                             fLog.flush()

                     if(debug):
                        stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"change value is not greater than 0. I am breaking out. \n")
                     break

        #flatten the list of structures that are created for the new iteration and update list of current structures
        listOfStructsCurr = [item for sublist in LOS_newLevel for item in sublist]

    return pdbFileNum, filesGen,listOfStructsAll,skipInfoAll, skipValAll

