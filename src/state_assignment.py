import os
import sys
import Bio
import numpy as np
import code
import copy
from collections import Counter
from Bio.PDB import *

import setup_protein as stp
import my_residue_atom as mra
import my_constants as mc
import hydrogen_placement_sp2 as hsp2
import hydrogen_placement_sp3 as hsp3
import close_atoms as cats



def setup_HIS(HISres, structure, HIStype = None, debug =0):

    ''' 
        Objective: To create a new type(HID/HIE/HIP) of HIS in the given structure and output the  original structure with HIS replaced with HID/HIE/HIP
        Description: Two things are done: 1. Name is fixed from HIS-->HIE/HID/HIP and 2. Add the appropriate hydrogens

        I/P: HISres: given a HIS residue, structure: where the new HIS type will belong to,
             HIStype: what are you creating either HIE/HIP/HID
        O/P: a dictionary where the structure with new HIE/HIP/HID resides
        '''
    ##This input structure is a parser that should already have been formed!
    ##Get structure, modelID, chainID,resID using res.fullID
    dict_struct = {} #Initialize a dictionary to output 
    structName = HIStype 
    dict_struct[structName] = structure
    modelID = HISres.parent.parent.id
    chainID = HISres.parent.id
    resID = HISres.id

    #Setup the name. ##HIStype should be "HIE" or "HID" or "HIP".  by over writing it!!
    dict_struct[structName][modelID][chainID][resID].resname = HIStype
    ####Add the hydrogens
    HIS_dict={'HIP':hsp2.place_hydrogens_HIP, 'HIE':hsp2.place_hydrogens_HIE,'HID':hsp2.place_hydrogens_HID}
    lastSerial = list(dict_struct[structName].get_residues())[-1].child_list[-1].serial_number
    lastSerial, hCoord = HIS_dict[HIStype](dict_struct[structName][modelID][chainID][resID], lastSerial)

    LP_dict = { 'HIE': [mc.hvysForLPsHIE, mc.LPSCnamesHIE], 'HID': [mc.hvysForLPsHID, mc.LPSCnamesHID]}
    try:
        hvys = LP_dict[HIStype][0]
        LPnameAll = LP_dict[HIStype][1]

        lastSerial, lpCoord = hsp2.place_lonepair(dict_struct[structName][modelID][chainID][resID], lastSerial, hvys, LPnameAll, debug=debug)
    except KeyError:
        stp.append_to_log(f"No Lone pairs to place for HIStype:{HIStype}, and unknown residue: {HISres} on chain: {HISres.parent}", debug=0 )
        if(debug==1):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"No Lone pairs to place for HIStype:{HIStype}, and unknown residue: {HISres} on chain: {HISres.parent}", debug=0)
        pass

    
    return dict_struct[structName]


def setup_rotamer_in_structure(res, structure, placeHyd = True, placeLP = True, debug = 0):
    '''
        Objective: To create a rotamer of a given residue in a given structure
        Description: In this function 1.remove any hydrogens/LP and then  
                    2.temporarily store original coords that need to be swapped
                    3.Add hydrogens and LPS

        I/P: res, and structure where the new rotomer will be placed
        O/P: structure with the residue rotamer present in it
    '''
    #step1: Remove any hydrogens and LPS res.detach('H')
    #step2: Switch and coords
    #step3: Add hydrogens like it should be

####Basic info regarding residue:
    modelID = res.parent.parent.id
    chainID = res.parent.id
    resID = res.id
    resHIS = res.resname in ['HIP', 'HIE','HID'] 
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
            stp.append_to_log(f"Tried to remove hydrogens to setup a rotamer but no hydrogens were present for: {res} and chain: {res.parent}\n", debug=0 )
            if(debug==1):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Tried to remove hydrogens to setup a rotamer but no hydrogens were present for: {res} and chain: {res.parent}\n", debug=0)
##############################step 2: Temp store original atom coords################################
    ##store original
    origAtoms = []
    origAtomsCoords = []
    for count,at in enumerate(dictSwitch[res.resname]):
        origAtoms.append(res[at])
        origAtomsCoords.append(res[at].get_coord())
        
#############################step3: Swap atom coords########################
    resInStruct = structure[modelID][chainID][resID]

###SWAPPING VALS:
    for i in range(len(origAtoms)):
        j=i+1 if(i%2==0) else i-1 #for even we are adding and odd we are subtracting
        resInStruct[origAtoms[i].id].set_coord(origAtomsCoords[j]  )
    
##############STEP4: If needed: Add Hydrogens and LPs!!
    if(placeHyd):
        lastSerial = list(structure.get_residues())[-1].child_list[-1].serial_number
        try:
            lastSerial, hCoord = rotomerPlaceH[resInStruct.resname](resInStruct, lastSerial, debug=debug)
        except KeyError:
            stp.append_to_log(f"Tried to place hydrogens to set up a rotamer but no hydrogens for: {res} on chain: {res.parent} \n", debug=0 )
            if(debug==1):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Tried to place hydrogens to set up a rotamer but no hydrogens for: {res} on chain: {res.parent} \n", debug=0)
            pass

    LP_dict = {'ASP': [mc.hvysForLPsASP, mc.LPSCnamesASP], 'GLU': [mc.hvysForLPsGLU, mc.LPSCnamesGLU],
                'ASN': [mc.hvysForLPsASN, mc.LPSCnamesASN], 'GLN': [mc.hvysForLPsGLN, mc.LPSCnamesGLN],
                'HIE': [mc.hvysForLPsHIE, mc.LPSCnamesHIE], 'HID': [mc.hvysForLPsHID, mc.LPSCnamesHID]
    }


    if(placeLP):
        try:
            hvys = LP_dict[res.resname][0]
            LPnameAll = LP_dict[res.resname][1]
            lastSerial, lpCoord = hsp2.place_lonepair(res, lastSerial, hvys, LPnameAll, debug=debug)
        except KeyError:
            stp.append_to_log(f"Tried to place lone pairs but no Lone pairs for: {res} on chain: {res.parent} \n", debug=0 )
            if(debug==1):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f" Tried to place lone pairs but no Lone pairs for: {res} on chain: {res.parent} \n", debug=0)
            pass

    resInStruct.isRotamer = 1

    return structure



def create_GLN_ASN_states(unRes, structure, debug=0):
    '''
       Objective: Create GLN rotamer or ASN rotamer configurations 
       Input: unknown residue and the structure the residue belongs to
       Output: structDict: A structure with original config  and Rotamer config og GLN/ASN

            '''

    #Variable requirement
    structDict = {} ##Dict in which output is stored!!
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
    structDict["struct"+str(unRID)+unRName+"R"] = setup_rotamer_in_structure(res2Rotamer, structure2rotomer, debug=debug)

    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"created all states for {unRes} of {unRes.parent} and added as a dictionary to: {structDict}", debug=0)

    return structDict


def create_HIS_states(unRes, structure, debug=0):

    ''' 
        Objective: Create all different configs of HIS:
                    1. HID
                    2. HIE
                    3. HIP
                    4. HID_rotomer
                    5. HIE_rotomer
                    6. HIP_rotomer
        output:  all the structs are stores as a list
        I/P: the unknown residue of HIS that needs all configs and 
            the structure that is of concern

        '''
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
        structDict[skey] = setup_HIS(unRes , structuresHIS[count], HIStype = name, debug=debug)
        skeyList.append(skey)
    
    ##AT this point: Structures with HIP,HIE and HID are formed.    
    ###Three structures of HIP, HIE, HID are:  structDict[skeyList[0]], structDict[skeyList[1]] structDict[skeyList[2]]
    ##now u want rotomer of HIE AND HID-so use the appropriate STRUCTS!!
    structureHIProtomer = structDict['struct'+str(unRID)+'HIP'].copy()
    structureHIErotomer = structDict['struct'+str(unRID)+'HIE'].copy()
    structureHIDrotomer = structDict['struct'+str(unRID)+'HID'].copy()

    ##THe rotomer of hie you want to convert should be pulled from the appropriate STRUCT-not the original one!!
    ###hie2 is the HIE residue that you would like to convert to hie2 rotomer
    hip2Rotamer = structureHIProtomer[modelID][chainID][unFullRID]
    hie2Rotamer = structureHIErotomer[modelID][chainID][unFullRID]
    hid2Rotamer = structureHIDrotomer[modelID][chainID][unFullRID]
            
    skey = "struct"+str(unRID)+"HIER" 
    skeyList.append(skey) 
    structDict[skey] = setup_rotamer_in_structure( hie2Rotamer, structureHIErotomer, debug=debug)


    skey = "struct"+str(unRID)+"HIDR"
    skeyList.append(skey)
    structDict[skey] = setup_rotamer_in_structure( hid2Rotamer, structureHIDrotomer, debug=debug) 
    skey = "struct"+str(unRID)+"HIPR"
    skeyList.append(skey)
    structDict[skey] = setup_rotamer_in_structure( hip2Rotamer, structureHIProtomer, debug=debug)

    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"created all states for {unRes} of {unRes.parent} and added as a dictionary to: {structDict}", debug=0)
    
    return structDict


def create_ASH(structForASH, ASP_A, ASP_B, LPs, debug=0):

    '''
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
    
    '''
    

   ##Create ASH structure-given two ASPs
    #ASH is a protonated ASP!
    ##In this function we are looking to add the hydrogen wrt to one oxygen of one ASP (ASP-A)
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
   ##Make sure the newly added hydrogen is attached to OD2!
   ##so incase we were looking at OD1-> we swap it with OD2!!
   ##We are adding "HD2" on ASP_A
    

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
    
    ##pick the one which has minimum distance, and the associate lonepair which we will replace with HD2
    minDist = min(list2check)
    ind = np.where(list2check == minDist)[0][0]
    lpChange = lpList[ind]
    lpChangeCoord = lpChange.coord

    stp.append_to_log(f"Need to create ASH from ASP_A:{ASP_A} from {ASP_A.parent}, and {ASP_B} from from {ASP_B.parent} with minimum distance: {minDist}.\n \
    All distances from lone pair(LP3/LP4 or LP5/LP6) associated with OD of ASP_A to OD of ASP_B: {dist_btw} are: {list2check}. Dropping LP:{lpChange} as it is associated with minimum distance \n", debug=0)

    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Need to create ASH from ASP_A:{ASP_A} from {ASP_A.parent}, and {ASP_B} from {ASP_B.parent} with minimum distance: {minDist}.\n \
    All distances from lone pair(LP3/LP4 or LP5/LP6) associated with OD of ASP_A to OD of ASP_B: {dist_btw} are: {list2check}. Dropping LP:{lpChange} as it is associated with minimum distance \n", debug=0)
    
    ##detatch that lonepair and add HD2 with the lonepair coords
    ASP_A.detach_child(lpChange.name)

    lastSerial = list(structASH.get_residues())[-1].child_list[-1].serial_number
    ASP_A.add(Bio.PDB.Atom.Atom(name='HD2', coord=lpChangeCoord, bfactor=0., occupancy=1., altloc=' ', fullname='HD2', serial_number=lastSerial+1,element='H'))
    structASH[modelID_A][chainID_A][ASP_A.id[1]].resname = 'ASH'

##########################Make sure OD2 is the one bonded with HD2!!
    ##if LPchange is LP5/LP6 then we are looking at OD2-so no change needed
    #But if LPchange is LP3/LP4: then we swap!
    #Also assuring ASH will have lonepairs: LP3,LP4,LP5. NO LP6 is present!!

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


def create_GLH(structForGLH, GLU_A, GLU_B, LPs, debug=0):

    '''
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
    
    '''
   ##Create GLH structure-given two GLUs
    #GLH is a protonated GLU!
    ##In this function we are looking to add the hydrogen wrt to one oxygen of one GLU (GLU-A)
    ##the oxygen wrt which HE2 is added depends on th input Lps:
    #If input LPs are:[LP3, LP4] then it is being added to OE1
    ##If input LPs are [LP5, LP5] then it is being added to OE2
    
   ##compute the minimum distance with respect to each oxygen
   ##Say we are on OE1, OE1 has 2 positions to which HE2 can be added (as it is sp2)
    #compute distance of LP3 wrt resB-OE1, and resB-OE2
    #compute distance of LP4 wrt resB-OE1 & res resB-OE2
    #In total we have 4 distance: LP3 wrt (GLU-B-OE1),
    #                             LP3 wrt (GLU-B-OE2)
    #                             LP4 wrt (GLU-B-OE1)
    #                             LP4 wrt (GLU-B-OE2)
   ##Add the hydrogen at that position where the distance is minimum
   ##Make sure the newly added hydrogen is attached to OE2!
   ##so incase we were looking at OE1-> we swap it with OE2!!
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
    
    ##pick the one which has minimum distance, and the associate lonepair which we will replace with HD2
    minDist = min(list2check)
    ind = np.where(list2check == minDist)[0][0]
    lpChange = lpList[ind]
    lpChangeCoord = lpChange.coord

    stp.append_to_log(f"Need to create GLH from GLU_A:{GLU_A} from {GLU_A.parent}, and {GLU_B} from {GLU_B.parent} with minimum distance: {minDist}.\n \
    All distances from lone pair(LP3/LP4 or LP5/LP6) associated with OE of GLU_A to OE of GLU_B: {dist_btw} are: {list2check}. Dropping LP:{lpChange} as it is associated with minimum distance \n", debug=0)

    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Need to create GLH from GLU_A:{GLU_A} from {GLU_A.parent}, and {GLU_B} from from {GLU_B.parent} with minimum distance: {minDist}.\n \
    All distances from lone pair(LP3/LP4 or LP5/LP6) associated with OE of GLU_A to OE of GLU_B: {dist_btw} are: {list2check}. Dropping LP:{lpChange} as it is associated with minimum distance \n", debug=0)

    
    ##detatch that lonepair and add HE2 with the lonepair coords
    GLU_A.detach_child(lpChange.name)

    lastSerial = list(structGLH.get_residues())[-1].child_list[-1].serial_number
    GLU_A.add(Bio.PDB.Atom.Atom(name='HE2', coord=lpChangeCoord, bfactor=0., occupancy=1., altloc=' ', fullname='HE2', serial_number=lastSerial+1,element='H'))
    structGLH[modelID_A][chainID_A][GLU_A.id].resname = 'GLH'

##########################Make sure OD2 is the one bonded with HD2!!

    ##if LPchange is LP5/LP6 then we are looking at OD2-so no change needed
    #But if LPchange is LP3/LP4: then we swap!
    #Also assuring ASH will have lonepairs: LP3,LP4,LP5. No LP6 is present!!

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


def create_ASH_states(ASP_A_orig, ASP_B_orig, structure, debug=0):
    
    ''' 
    objective: To create four different ASP states given two ASP in h-bonding distance with each other 

    I/P: The two different ASPs ID and the structure that needs to be updates
    
    O/P: The four different structures in dict data type
    '''
    ##When two ASP's are in h-bonding distance of each other, one ASH is formed.
    ##Say there are two ASPs: APS-A, and ASP-B. There are two oxygen on each ASP
    ##So there are total of 4 potential states for ASH( each oxygen can potentially get a hydrogen)
    ##step1: Determine the 4 states!!
    ##Question: Since each oxygen is sp2 and has two potential positions for hyrogen-find the hydrogen position 
    ##for each of the four states
    ##Then one of the side chain oxygen
    
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
        structDict[sNames[i-1]] = create_ASH(structForASH, ASP_list[i-1], ASP_list[-i], LPnames[(i-1)%2], debug=debug)

    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"For ASPs: ASP_A:{ASP_A} of {ASP_A.parent} and ASP_B: {ASP_B.parent}, the ASH structures are created and stored in a dictionary: {structDict}")
    return structDict


def create_GLH_states(GLU_A_orig, GLU_B_orig, structure, debug=0):

    
    ''' 
    objective: To create four different GLU states given two GLU in h-bonding distance with each other 

    I/P: The two different GLUs ID and the structure that needs to be updates
    
    O/P: The four different structures in dict data type
    '''
    ##When two GLU's are in h-bonding distance of each other, one GLH is formed.
    ##Say there are two GLUs: GLU-A, and GLU-B. There are two oxygen on each GLU
    ##So there are total of 4 potential states for GLH( each oxygen can potentially get a hydrogen)
    ##Step1: Determine the 4 states!!
    ##Question: Since each oxygen is sp2 and has two potential positions for hyrogen-find the hydrogen position 
    ##for each of the four states
    ##Then one of the side chain oxyge
    
    

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
        structDict[sNames[i-1]] = create_GLH(structForGLH, GLU1[i-1], GLU1[-i], LPnames[(i-1)%2], debug=debug)


    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"For ASPs: ASP_A:{ASP_A} of {ASP_A.parent} and ASP_B: {ASP_B.parent}, the ASH structures are created and stored in a dictionary: {structDict}")

    return structDict


def get_ASP_GLU_pair(currASP_GLU, structure, ASP_GLUdist, debug=0):

    '''
    objective: For a given ASP/GLU (any of the two oxygen side chain atoms),
               find another ASP/GLU (any of the two oxygen side chain atoms)
               in the h-bonding distance of each other

    input: -Given ASP/GLU for which we are looking for another ASP/GLU in its hbonding distance
           -structure: the structure you want to use to hunt for the other ASP/GLU (you may not need to pass this-just use 
                   currASPs parent?)
           -ASP_GLUdist: the permissible hbonding distance

    output: the pair ASP_GLU who in the the hbonding distance within currASP_GLU

    '''
    fLogName = stp.get_log_file_name(debug=0)
    fLog = open(fLogName, "a")

    if(debug==1):
        stp.start_debug_file(__name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debug_file_name(debug=0)
        fd = open(fDebugName, "a")

    unknownASP_GLUatom = []
    unknownASP_GLUatomInfo = []

    ##Use the list of ALL ASPs to find ASP partner:
    if(currASP_GLU.resname == 'ASP'):
        searchASP_GLUatoms = stp.get_all_unknown_ASP_OD_atoms(structure, debug=debug)
        oxygenName1 = 'OD1'
        oxygenName2 = 'OD2'
    else:
        searchASP_GLUatoms = stp.get_all_unknown_ASP_OE_atoms(structure, debug=debug)
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

    
    if(debug==1):
        fd.write(f"Atom found in hbonding distance neighborhood of {oxygenName1} belonging to {currASP_GLU} of {currASP_GLU.parent} is : {potUnknownAtom1}, and\n \
 Atom found in hbonding distance neighborhood of {oxygenName2} belonging to {currASP_GLU} of {currASP_GLU.parent} is : {potUnknownAtom2} \n")


        fd.write(f"\n For both oxygen atoms the currrent ASP/GLU oxygen atom, current ASP/GLU oxygens residue, current atom residue chain, uat(atom belonging to unknown ASP/GLU in hbonding dist), uat residue, uats residues chain, distance between oxygen atom of current ASP/GLU and uat are the following:\n \
                {unknownASP_GLUatomInfo}\n\n ")
        fd.flush()

    fLog.write(f"Atom found in hbonding distance neighborhood of {oxygenName1} belonging to {currASP_GLU} of {currASP_GLU.parent} is : {potUnknownAtom1},\n \
            and  Atom found in hbonding distance neighborhood of {oxygenName2} belonging to {currASP_GLU} of {currASP_GLU.parent} is : {potUnknownAtom2} \n")
    fLog.flush()
   
    unknownASP_GLUlist = []

    #In the list od unknownASP atom that was just created-find its parent and append it
    for atom in unknownASP_GLUatom:
        unknownASP_GLUlist.append(atom.parent)
    
    #create a set so we can have the unique ASP residues
    unknownASP_GLUset = set(unknownASP_GLUlist)
    #since this code is meant for only diad! So PLS confirm you get only one other ASP!! 
    ##otherwise exit the program
    #Next line should ever happen-because if ASP is > diad then- we dont consider it as unknowns-user needs to worry about it!
    if(len(unknownASP_GLUset)> mc.numASPsGLUs-1):
        fLog.write(f"number of unique ASPs: {len(unknownASP_GLUset)}. Please check\n\n")
        fLog.flush()
        
        fd.write(f"number of unique ASPs: {len(unknownASP_GLUset)}. Please check\n\n")
        fd.flush()

    
    
    if(not unknownASP_GLUset):
        unknownASP_GLUpair = None
    else: 
        unknownASP_GLUpair = list(unknownASP_GLUset)[0]
        
        for unknownASP_GLU in unknownASP_GLUset:
            fLog.write(f"The unknownASP/GLU pair for {currASP_GLU} of {currASP_GLU.parent} is:  {unknownASP_GLU} of chain: {unknownASP_GLU.parent}\n")
            fLog.flush()
            if(debug==1):
                fd.write(f"The unknownASP/GLU pair for {currASP_GLU} of {currASP_GLU.parent} is:  {unknownASP_GLU} of chain: {unknownASP_GLU.parent}\n")
                fd.flush()

    if(debug==1):
        stp.end_debug_file(__name__,sys._getframe().f_code.co_name, fd)
    fLog.close()

    return unknownASP_GLUpair


def energy_of_donor_for_all_close_atoms(hvyAt, allCloseAtoms, chV_level,con, debug=0):
    '''
    Objective: To compute energy interaction with neighboring atoms when my reference atom(current atom part of the unknown res) is a donor.
        I/P: hvyAt: heavy atom which is the part of unknown residue, allCloseAtoms: All close atoms we are considering
             con: name of the state, for example: GLN or GLNR or ASH_OD1A
             chV_level: to track and debug 
        O/P: enValList: Energy value list, and enSumTotal:  current total sum of the energy
 
    '''

    fLogName = stp.get_log_file_name(debug=0)
    fLog = open(fLogName, "a")

    if(debug==1):
        stp.start_debug_file( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debug_file_name(debug=0)
        fd = open(fDebugName, "a")
        myHvyAt = mra.my_atom(hvyAt)
        myHvyAtBehav = myHvyAt.get_behavior().abbrev
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
        stp.end_debug_file(__name__,sys._getframe().f_code.co_name, fd) 
    
    enSumTotal = 0
    enValList = []

    ####If SP3 side chain is unknown and I hit a sp3. Add Hydrogens and LP based on name of residue
    struct = hvyAt.parent.parent.parent.parent  ###Since only one struct is being used. Lets usee hvy at struc
    modelIDhvyAt = hvyAt.parent.parent.parent.id
    chainIDhvyAt = hvyAt.parent.parent.id
   
    ##Iterating over all close atoms, and implement energy computation depending on close atom behavior 
    #energy computation is done, since heavy atom behavior is already known-Its a DONOR!
    for i in range(1, np.shape(allCloseAtoms)[0]):
         
        currCloseAtom = allCloseAtoms[i][0]
        closeAtomBehav = mra.my_atom(currCloseAtom).get_behavior().abbrev
        currCloseResID = currCloseAtom.parent.id
        modelIDcloseAt = currCloseAtom.parent.parent.parent.id
        chainIDcloseAt = currCloseAtom.parent.parent.id
        currCloseRes = struct[modelIDcloseAt][chainIDcloseAt][currCloseResID] 
        ##Add the hydrogen id close atom is sp3
        if(currCloseRes.isSCHknown == 0):
            #This is so that the hvy atom parent is assumed to be known while computing SER close atoms. This flag is turned to 0 and after the computation
            hvyAt.parent.isKnown = 1
            lastSerial = list(struct.get_residues())[-1].child_list[-1].serial_number

            if(currCloseRes.resname=='SER' or currCloseRes.resname=='THR'):
                lastSerial, hLPCoords = hsp3.place_hydrogens_lonepairs_SER_THR(currCloseAtom.parent, lastSerial, debug=debug) 
                if(hLPCoords == []):
                    currCloseRes.isSCHknown = 0
                else:
                    currCloseRes.isSCHknown = 1

                hvyAt.parent.isKnown = 0

            elif(currCloseRes.resname == 'LYS'):
                lastSerial, hLPCoords = hsp3.place_hydrogens_lonepairs_LYS(currCloseAtom.parent, lastSerial, debug=debug)
                if(hLPCoords == []):
                    currCloseRes.isSCHknown = 0
                else:
                    currCloseRes.isSCHknown = 1
                hvyAt.parent.isKnown = 0
            elif(currCloseRes.resname == 'TYR'):
                lastSerial, hLPCoords = hsp2.place_hydrogens_TYR(currCloseAtom.parent, lastSerial, debug=debug)
                if(hLPCoords == []):
                    currCloseRes.isSCHknown = 0
                else:
                    currCloseRes.isSCHknown = 1
                hvyAt.parent.isKnown = 0
            else:
                continue
        #After attempting to place hydrogens/LP-if there is no known closeAtoms-then pls continue and try again in another iteration if it comes up!
        if(currCloseRes.isSCHknown==0): 
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
            fLog.write(f"No backbone hydrogen is present for energy computations. Heavy atom: {hvyAt} belongs to {hvyAt.parent} of its chain: {hvyAt.parent.parent}. Perhaps its the first residue on the chain. \n")
            fLog.flush()

        else:
            atomHs = cats.get_hydrogen_connected_to_donor(hvyAt, debug=debug)
            #going over all the hyrogen atoms connected to the heavy atom
            for hat in atomHs:
                if(closeAtomBehav=='do'):
                    enVal, enSum=cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive=0, atype='SP2', hName=hat.id, chV_levelVal=chV_level, debug=debug)
                    enSumTotal = enSumTotal+enSum
                    enValList.append(enVal)

                elif(closeAtomBehav=='ac'):
                    #Next if it is an acceptor
                    enVal, enSum = cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive=1, atype='SP2',hName=hat.id, chV_levelVal=chV_level, debug=debug)
                    enSumTotal = enSumTotal+enSum
                    enValList.append(enVal)

                elif(closeAtomBehav=='bo'):
                    #If both then-First acceptor
                    enVal0, enSum0 = cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive = 1, atype = 'SP2', hName = hat.id, chV_levelVal = chV_level, debug=debug)
                    #Next donor
                    enVal1, enSum1 = cats.compute_energy_as_donor(currCloseAtom, hat.coord, hvyAt, attractive = 0, atype = 'SP2', hName = hat.id, chV_levelVal = chV_level, debug=debug)
                    enSumTotal = enSumTotal+enSum0+enSum1
                    enValList.append([enVal0, enVal1])
                    
                else:
                    continue
    fLog.close()
    return enValList,enSumTotal



def energy_of_acceptor_for_all_close_atoms(hvyAt, allCloseAtoms, chV_level,con, debug=0):

    '''
    Objective: To compute energy interaction with neighboring atoms when my reference atom(current atom part of the unknown res) is an ACCEPTOR.
        I/P: hvyAt: heavy atom which is the part of unknown residue, allCloseAtoms: All close atoms we are considering
            chV_level: to track and debug, con: name of the residue state -to track and debug
        O/P: enValList: Energy value list, and enSumTotal:  current total sum of the energy
 
        '''

    enSumTotal=0
    enValList=[]
    myHvyAt=mra.my_atom(hvyAt)
    LPAtoms=myHvyAt.get_lonepairs_atoms()

    struct=hvyAt.parent.parent.parent.parent  ###Since only one struct is being used. Lets usee hvy at struc
    modelIDhvyAt=hvyAt.parent.parent.parent.id
    chainIDhvyAt=hvyAt.parent.parent.id
    
    if(debug==1):
        stp.start_debug_file( __name__, sys._getframe().f_code.co_name)
        fDebugName=stp.get_debug_file_name(debug=0)
        fd = open(fDebugName, "a")
        myHvyAt = mra.my_atom(hvyAt)
        myHvyAtBehav = myHvyAt.get_behavior().abbrev

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
        stp.end_debug_file(__name__,sys._getframe().f_code.co_name, fd) 

    for i in range(1, np.shape(allCloseAtoms)[0]):
        currCloseAtom = allCloseAtoms[i][0]
        closeAtomBehav = mra.my_atom(currCloseAtom).get_behavior().abbrev

        currCloseResID = currCloseAtom.parent.id
        modelIDcloseAt = currCloseAtom.parent.parent.parent.id
        chainIDcloseAt = currCloseAtom.parent.parent.id

        currCloseRes = struct[modelIDcloseAt][chainIDcloseAt][currCloseResID]

        #Add the hydrogen if close atom is sp3!!
        if(currCloseRes.isSCHknown == 0):
            #This is so that the hvy atom parent is assumed to be known while computing SER close atoms. This flag is turned to 0 and after the computation
            hvyAt.parent.isKnown=1
            lastSerial=list(struct.get_residues())[-1].child_list[-1].serial_number
            if(currCloseRes.resname == 'SER' or currCloseRes.resname == 'THR'):
                lastSerial, hLPCoords=hsp3.place_hydrogens_lonepairs_SER_THR(currCloseAtom.parent, lastSerial, debug=debug)
                hvyAt.parent.isKnown=0
                if(hLPCoords == []):
                    currCloseRes.isSCHknown=0
                else:
                    currCloseRes.isSCHknown=1

            elif(currCloseRes.resname=='LYS'):
                lastSerial, hLPCoords=hsp3.place_hydrogens_lonepairs_LYS(currCloseAtom.parent, lastSerial, debug=debug)
                
                if(hLPCoords==[]):
                    currCloseRes.isSCHknown=0
                else:
                    currCloseRes.isSCHknown=1
                hvyAt.parent.isKnown=0

            elif(currCloseRes.resname=='TYR'):
                lastSerial, hLPCoords=hsp2.place_hydrogens_TYR(currCloseAtom.parent, lastSerial, debug=debug)
                if(hLPCoords==[]):
                    currCloseRes.isSCHknown=0
                else:
                    currCloseRes.isSCHknown=1
                hvyAt.parent.isKnown=0

            else:
                continue
    
        #After attempting to place hydrogens/LP-if there is no known closeAtoms-then pls continue and try again in another iteration if it comes up!

        if(currCloseRes.isSCHknown==0): 
            stp.append_to_log(f"Current Close atom:{currCloseRes} is still unknown! Going on to the next close atom \n", debug=0)
            continue
        hvyAt.parent.isKnown=0
        #Going over all the lone pair atoms for acceptors. All acceptors have lone pairs in-order to accept a H atom
        for j in range(len(LPAtoms)):
                lp_vec = LPAtoms[j].get_vector()

                if(closeAtomBehav == 'do'):
                    #If close atom is a donor
                    enVal,enSum = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 1, atype = 'SP2', chV_levelVal = chV_level, debug=debug)
                    #summing the total energy for a given heavy atom
                    enSumTotal = enSumTotal+enSum
                    enValList.append(enVal)                            
                elif(closeAtomBehav == 'ac'):
                    #If close atom is an acceptor
                    enVal, enSum = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 0, atype = 'SP2', chV_levelVal = chV_level, debug=debug)
                    #summing the total energy for a given heavy atom
                    enSumTotal=enSumTotal+enSum
                    enValList.append(enVal)

                elif(closeAtomBehav == 'bo'):
                    #If close atom is both then taking the donor first(attractive=1) and then acceptor(attractive=0)
                    #remember our hvy atom is still an acceptor!!
                    enVal0,enSum0 = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 1, atype = 'SP2', chV_levelVal = chV_level, debug=debug)
                    enVal1,enSum1 = cats.compute_energy_as_acceptor(hvyAt, lp_vec, currCloseAtom, attractive = 0, atype = 'SP2', chV_levelVal = chV_level, debug=debug)

                    #summing the total energy for a given heavy atom
                    enSumTotal=enSumTotal+enSum0+enSum1
                    enValList.append([enVal0, enVal1])
                else:   
                    continue
                

    return enValList,enSumTotal



def compute_energy_for_given_atoms(resState, givenAtoms, chV_level, con, debug=0):

    '''
    Objective: Given a configuration, compute energy sum of all the donors and acceptors present in the residue.
    I/P: resState: It is the unknown residue under consideration, givenAtoms: to consider
                  chV_level: to track and debug, con: name of the residue state -to track and debug

    O/P: enSumTotal: Total Energy sum of the configuration, 
         enSumForHvys: list of each heavy atom along with its energy sum

    '''
    fLogName = stp.get_log_file_name(debug=0)
    fLog = open(fLogName, "a")

    struct = resState.parent.parent.parent

    enSumTotal = 0
    enSumForHvys = []
   
    #Iterate over the given atoms in the input (as we need the energy for all those atoms wrt its close atoms)
    for hvyAt in givenAtoms:
        myHvyAt = mra.my_atom(hvyAt)
        myHvyAtBehav = myHvyAt.get_behavior().abbrev

        #create a custom list to find all close atoms
        customList = stp.get_known_donor_acceptor_list_for_one_atom(struct, hvyAt, aaType = 'DONOR_ACCEPTOR_BOTH', debug=debug)
        ##Get a list of all close Atoms (which are donor, acceptor, and both) for the given hvy atom
        allCloseAtoms = cats.get_all_close_atom_info_for_one_atom(hvyAt, customList, debug=debug)

        fLog.write(f"Evaluate given residue state, hvyAt: {hvyAt} and All close atoms: {allCloseAtoms} and enSumTotal is: {enSumTotal}\n")
        fLog.flush()
    

        if(myHvyAtBehav == 'do'):
            #If the heavy atom is a donor the compute energy for all close atoms
            enValList, enSumTotalDonor = energy_of_donor_for_all_close_atoms(hvyAt, allCloseAtoms, chV_level,con, debug=debug)
            enSumTotal = enSumTotal + enSumTotalDonor
            fLog.write(f"energy sum donor from func:{enSumTotalDonor} and enSumTotal:{enSumTotal} \n")
            fLog.flush()
        elif(myHvyAtBehav == 'ac'):
            #If heavyAtom is acceptor-compute energy for all its close atom. In the unknownRes-we do not have a situation
            #for BOTH so I do not consider it!!
            enValList, enSumTotalAcceptor = energy_of_acceptor_for_all_close_atoms(hvyAt, allCloseAtoms, chV_level,con, debug=debug) 
            enSumTotal = enSumTotal + enSumTotalAcceptor
            fLog.write(f"energy sum acceptor from func:{enSumTotalAcceptor} and enSumTotal:{enSumTotal}\n")
            fLog.flush()
                
        else:
            continue

        enSumForHvys.append([hvyAt, enSumTotal])


    fLog.write(f"For this state:{resState} tot energy is: {enSumTotal}\n")
    fLog.write("************************************************************************\n")
    fLog.write("************************************************************************\n")
    fLog.flush()

    
    fLog.close()
    return enSumTotal, enSumForHvys


def compute_energy_for_given_state(resState, chV_level, con, debug=0):

    '''
    Objective: Given a configuration, compute energy sum of all the donors and acceptors present in the residue.
    I/P: config: It is the unknown residue under consideration.
    O/P: enSumTotal: Total Energy sum of the configuration, 
         enSumForHvys: list of each heavy atom along with its energy sum
    '''

    #Find list side chain active atoms, i.e donor/acceptor
    LOAAresState = mra.my_residue(resState).get_unknown_residue_acceptor_donor_atoms() 
    enSumTotal = 0
    enSumForHvys = []
    #Find the energy for all the side chain active atoms of the unknown residue
    enSumTotal, enSumForHvys = compute_energy_for_given_atoms(resState, LOAAresState, chV_level, con, debug=debug)

    if(debug==1):
        stp.start_debug_file( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debug_file_name(debug=0)
        fd = open(fDebugName, "a")
        fd.write(f"For this state:{resState} (of {resState.parent}) where isRotamer: {resState.isRotamer}, given atoms: {LOAAresState} total energy is: {enSumTotal}, while the individual energies assembled as: heavy atom and energy value is: {enSumForHvys}\n")
        fd.write("************************************************************************\n")
        fd.write("************************************************************************\n")
        fd.flush() 
        stp.end_debug_file(__name__,sys._getframe().f_code.co_name, fd) 

    return enSumTotal, enSumForHvys


def compute_energy_for_all_states(unknownRes, structStates, chV_level, debug=0):

    ''' objective: compute energy for all states of the given unknown residue 
        I/P: the unknown residue, and all its states
        O/P: energy list of all residue stats, and states
    '''
    fLogName = stp.get_log_file_name(debug=0)
    fLog = open(fLogName, "a")
    resStates = []
    energyList = []
    #because the modelID and chainID for a given residue-and its different states remain the same (They will be in same structure/chain/model)
    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id

   

    ListOfSS = list(structStates)
    if(unknownRes.resname == 'ASP' or unknownRes.resname == 'GLU'):
        ##Get the 2 unknownresidue
        sUn = unknownRes.parent.parent.parent
        unknownASPpair = get_ASP_GLU_pair(unknownRes, sUn, mc.deltaD, debug=debug)
        unknownRes0 = [unknownRes, unknownRes, unknownASPpair, unknownASPpair]
    else:
        unknownRes0 = [unknownRes, unknownRes, unknownRes, unknownRes]

    #Going over all states of the unknown residue
    for count,con in enumerate(mc.unResDict[unknownRes.resname][0]):
        fLog.write("************************************************************************\n")
        #accesing the struct state:
        structState = structStates['struct'+str(unknownRes0[count].id[1])+con]
        #accesing the residue state:
        resState = structState[modelID][chainID][unknownRes0[count].id]
        fLog.write(f"State: {con}, with ID {unknownRes0[count].id[1]} and chain: {unknownRes0[count].parent} and isRotamer: {resState.isRotamer} \n")
        fLog.flush()
        if(debug==1):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f" ************************************************************************\n State: {con}, with ID {unknownRes0[count].id[1]} and chain: {unknownRes0[count].parent} and isRotamer: {resState.isRotamer} \n")
        resStates.append(resState)
        #compute energy for the given state:
        enSumTotalResState, enSumForHvys = compute_energy_for_given_state(resState, chV_level, con, debug=debug)
        energyList.append([resState, enSumTotalResState, structState, resState, con ])

    fLog.close()
    return energyList, resStates


def branch_structure(unknownRes, structure, debug=0):

    '''
    objective: to create branch of the structure provided. This function creates all the possible options
               Note: we do not convert it into known at this point as it is only done when setting a state. Here we just create branch to analyze the different rotameric/protonation states.
    Input: -unknownRes-the unknown residue you want to branch
           -structure it belongs to
    Output: S - a dict of the multiple structures

    '''

    #depending on the name of unknownRes we create the structures
    if(unknownRes.resname=='HIS'):      
        S = create_HIS_states(unknownRes, structure, debug=debug)
    elif(unknownRes.resname=='ASN' or unknownRes.resname == 'GLN'):  
        S = create_GLN_ASN_states(unknownRes,structure, debug=debug)
    elif(unknownRes.resname=='ASP' or unknownRes.resname == 'GLU'):
        resPair = get_ASP_GLU_pair(unknownRes, structure, mc.deltaD, debug=debug)
        if(unknownRes.resname=='ASP'):
            S = create_ASH_states(unknownRes, resPair, structure, debug=debug)
        else:
            S = create_GLH_states(unknownRes, resPair, structure, debug=debug)
    else:
        S = None
    
    if(debug==1):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"For the unknown residue: {unknownRes} of {unknownRes.parent}, the following structures are created and stored in a dictionary: {S}")
    return S



def set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, collectData, branchedS, MSG, debug=0):

    ''' 
    objective: To set the state in the given structure by converting it to known and removing it from the list of unknowns
        
    Input: -unknownRes: the unknown residue of concern
           -res2keep: the residue state you want to keep
           -nameOfS2keep: name of the structure you want to keep
           -unknownResMod: List of unknownRes modified-will remove the unknown Res from it as it is now known
           -changeVal: number of time changes occur
           -collectData: appending info about the new known to the data structure 
           -branchedS: the branched structure dict you want to use to assign the value to be known
           -MSG: The message you want to print on screen while setting the state!!
    
    Output: new and updated structure with known residue, updated list of unknown residue, changeVal and collect Data
    '''
    struct = branchedS[nameOfS2keep]

    fLogName = stp.get_log_file_name(debug=0)
    fLog = open(fLogName, "a")
    fLog.write(MSG+"\n")
    fLog.flush()
    
    collectData.append([unknownRes, res2keep, unknownRes.parent.id, MSG ]) 
    changeVal+=1
    
    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id
    struct[modelID][chainID][unknownRes.id].isKnown = 1

    if(unknownRes.resname == 'ASP' or unknownRes.resname == 'GLU'):
        sUn = unknownRes.parent.parent.parent
        unknownASP_GLUpair = get_ASP_GLU_pair(unknownRes, sUn, mc.deltaD, debug=debug)
        modelIDpair = unknownASP_GLUpair.parent.parent.id
        chainIDpair = unknownASP_GLUpair.parent.id
        struct[modelIDpair][chainIDpair][unknownASP_GLUpair.id].isKnown = 1

    unknownResMod.remove(unknownRes)
    structure = copy.deepcopy(branchedS[nameOfS2keep])

    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f" setting state for: {unknownRes} of {unknownRes.parent} \n \
                 residue kept:{res2keep} of {res2keep.parent}, isRotamer:{res2keep.isRotamer},\n \
                 name of structure kept:{nameOfS2keep},\n \
                 new list of unknowns: {unknownResMod},\n \
                 current change value:{changeVal},\n\
                 additional info inclued:[unknownRes, res2keep, unknownRes.parent.id, MSG ]:{collectData},\n \
                 branched structure is:{branchedS},\n \
                 updated structure kept is: {structure}\n\
                 message to print:{MSG}\n ")

    fLog.close()

    return structure, unknownResMod, changeVal, collectData


def get_structure_name(res, moreInfo=None, debug=0):
    
    '''
    objective: to get structure name-specifically by checking if the unknown residue it is a rotomer or not

    Input:  -res: is the residue to check
            -moreInfo: name of res describing the unknown res. For example, "HIER" is HIE in rotameric state(non orig state).
            It will describe the name of structure as 'struct123HIER' assuming 123 is residue number of HIE in consideration.

    Output: nameOfStruct: -name of the struct
    
    '''
    nameOfStruct = 'struct'+str(res.id[1]) + moreInfo

    return nameOfStruct


def get_degenerate_structure_names(degenArray, debug=0):


    '''
    objective: To extract all the structure names for the degenerate cases using the dgenArray data type

    Input: degenArray:- It is the data type which contains information for the degenCases

    Output: Names of all structs that is extracted from degenArray

    '''
    #list of degen struct names for output
    degenStructNames = []
    ##degenArray has 2 columns. Zeroth col:-res and first col has energy Val
    ##degenArray's zeroth column has all residues. Iterating over and extracting the names by using get_structure_name
    for count, res in enumerate(degenArray[:,0]): 
        nameOfStruct = get_structure_name(res, moreInfo = degenArray[count,-1], debug=debug)
        degenStructNames.append(nameOfStruct)
    
    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"The degenerate structure names are: {degenStructNames}\n And more details are(resState, enSumTotalResState, structState, resState, con/name of the state ): {degenArray}\n")

    return degenStructNames


def get_HIP_energies(resHIP, resHIPR, chV_level, debug=0):

    '''
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

    '''

    con = 'OS'#original structure
    enSumTotND1, enSumForND1 = compute_energy_for_given_atoms(resHIP, [resHIP['ND1']], chV_level,con, debug=debug)
    enSumTotNE2, enSumForNE2 = compute_energy_for_given_atoms(resHIP, [resHIP['NE2']], chV_level,con, debug=debug)

    con = 'RS'#rotamer structure
    enSumTotND1Ro, enSumForND1Ro = compute_energy_for_given_atoms(resHIPR, [resHIPR['ND1']], chV_level,con, debug=debug)
    enSumTotNE2Ro, enSumForNE2Ro = compute_energy_for_given_atoms(resHIPR, [resHIPR['NE2']], chV_level,con, debug=debug)
    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"computing energy value for HIPs: {resHIP} and {resHIPR} .The energies for the original state are the following.\n \
  For ND1, total sum and individual values are: {enSumTotND1}, {enSumForND1} \n \
  For NE2, total sum and individual values are: {enSumTotNE2}, {enSumForNE2} \n\n \
The energies for the rotamer state are the following.\n \
  For ND1 in rotamer state, total sum and individual values are: {enSumTotND1Ro}, {enSumForND1Ro} \n \
  For NE2 in rotamer state, total sum and individual values are: {enSumTotNE2Ro}, {enSumForNE2Ro} \n \
                ", debug=0)

    return enSumTotND1, enSumTotNE2, enSumTotND1Ro, enSumTotNE2Ro



def evaluate_degenerate_cases( unknownRes,structure, S, energyArray, sortedEnArray, unknownResMod, changeVal, skipVal, collectData, skipResInfo, debug = 0):

    '''
    objective: to update structures for the potential degenerate cases. By def: a degerate case is 
                where the two or more different states of a residue have energy difference less than 1 Kcal.
    
    Input: -unknownRes: the unknown residue with respect to which the degenerate case is developed
           -structure: the structure that needs to be updated.
           -S: it is a dictionary of multiple structures of the multiple degenerate cases.
           -LOCA_unknownRes: List of Close Atoms of the unknown residue
           ###TODO: Do we really need energy array-can we do with just sortedEnArray??
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

    '''
    fLogName = stp.get_log_file_name(debug=0)
    fLog = open(fLogName, "a")

    ##Creating 5 possible cases of not necessarily denergate cases:
    #1. all energy values are zero-not a degen case
    #2. all energy values are pos -DEGEN CASE
    #3. all energy values are negetive- DEGEN CASE
    #4. smallest energy value is zero- not a degen case. pick zero!
    #5. smallest energy value is negetive-not degen if the second smallest energy val=0 or >0, else Degen case

    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id

    sortedResNamesArr = np.array([x.resname for x in sortedEnArray[:,0]])
    ##Create conditionals for all possibilities
    allEnergyZero = all(c == 0 for c in energyArray[:,1])    ##All close atoms have zero energy
    allEnergyPos = all(c > 0 for c in energyArray[:,1])
    allEnergyNeg = all(c < 0 for c in energyArray[:,1])
    smallestEnergyZero = sortedEnArray[0,1]==0
    smallestEnergyNeg =sortedEnArray[0,1]<0

    ##There is a reason I am doing if else here-since the order matters! (thus not doing dict-try-except)
    if(allEnergyZero):
    ##NOT A DEGEN CASE
        #IF HIS->default to HIE else keep the original state
        if(unknownRes.resname=='HIS'): 
            nameOfRes2keep='HIE'
        else:
            nameOfRes2keep=unknownRes.resname

        if(unknownRes.resname=='GLU' or unknownRes.resname=='ASP'):
            del S
            struct = copy.deepcopy(structure)
            sUn = unknownRes.parent.parent.parent
            unknownASP_GLUpair = get_ASP_GLU_pair(unknownRes, sUn, mc.deltaD, debug=debug)
            modelID_pair = unknownASP_GLUpair.parent.parent.id
            chainID_pair = unknownASP_GLUpair.parent.id


            struct[modelID][chainID][unknownRes.id].isKnown=1
            struct[modelID_pair][chainID_pair][unknownASP_GLUpair.id].isKnown=1
            res2keep=unknownRes 
            MSG=f'{unknownRes} and pair: {unknownASP_GLUpair} All energies = 0  '
            fLog.write(MSG+"\n")
            fLog.flush()
            collectData.append([unknownRes, res2keep, unknownRes.parent.id, MSG ]) 
            unknownResMod.remove(unknownRes)
            unknownResMod.remove(unknownASP_GLUpair)
            structure = copy.deepcopy(struct)  
            changeVal +=1
            return structure, unknownResMod, changeVal, skipVal, collectData, skipResInfo

        nameOfS2keep = 'struct'+str(unknownRes.id[1])+nameOfRes2keep
        res2keep= S[nameOfS2keep][modelID][chainID][unknownRes.id]
        
        #min energy stored is the energy value picked depending on the residue name to keep
        ind = np.where(sortedResNamesArr[:]== res2keep.resname)[0][0]
        enValPick =sortedEnArray[ind][1]
       
        #string that will be printed out to the user along with other default info.
        condStr = "ALL ENERGIES = 0"

        msg2Usr = f"{condStr}, Saving residue: {res2keep} to residue state corresponding to minimum energy  i.e {enValPick} and isRotamer:{res2keep.isRotamer}"
        ###Setting the state-change value and skip values are updated in the function (set_state)
        structure, unknownResMod, changeVal, collectData = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, collectData, S, msg2Usr, debug=debug)

        del S

    elif(smallestEnergyZero):
####NOT A DEGENERATE CASE AT ALL
         res2keep = sortedEnArray[0,0] ##PICKING the smallest as 0<+v, -v<0

         moreInfoName = sortedEnArray[0,-1]
         nameOfS2keep = get_structure_name(res2keep, moreInfo = moreInfoName, debug=debug)
 
         #Min energy is the smallest in the sorted row 
         enValPick =sortedEnArray[0][1]  
         #string that will be printed out to the user along with other default info.
         condStr = "Smallest Energy is zero and the next one is either pos or zero!"
         msg2Usr = f"{condStr},Saving residue: {res2keep} to residue state corresponding to minimum energy  i.e {enValPick} and isRotamer:{res2keep.isRotamer} (specifically:smallest: {sortedEnArray[0,1]} and next smallest:{sortedEnArray[1,1]})"
         ###Setting the state-change value and skip values are updated in the function (set_state)
         structure, unknownResMod, changeVal, collectData = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, collectData, S, msg2Usr, debug=debug)

         del S

    elif(smallestEnergyNeg):
        #a DEGEN case if the second smallest is not zero or >0
        if(sortedEnArray[1,1]==0 or sortedEnArray[1,1]>0):
            #not degenerate case
            res2keep = sortedEnArray[0,0]

            moreInfoName = sortedEnArray[0,-1]
            nameOfS2keep = get_structure_name(res2keep, moreInfo = moreInfoName, debug=debug)
            #Min energy is the smallest in the sorted row 
            enValPick =sortedEnArray[0][1]
           
            #string that will be printed out to the user along with other default info.
            condStr = "Smallest Energy is NEGETIVE and the next pos or zero!"

            msg2Usr = f"{condStr}, Saving residue: {res2keep} to residue state corresponding to minimum energy  i.e {enValPick} and isRotamer:{res2keep.isRotamer}, (specifically:smallest: {sortedEnArray[0,1]} and next smallest:{sortedEnArray[1,1]})"
            ###Setting the state-change value and skip values are updated in the function (set_state)
            structure, unknownResMod, changeVal, collectData = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod,changeVal, collectData, S, msg2Usr, debug=debug)

            del S
        else:
            #A DEGEN CASE: as smallest and the second smallest energies are zero
            #In this particular case we do not know how many of the res states are degenerate
            #Create a degen list
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
            indDegen = np.where(EnDiffWithMin<1)
            #use the index where diff of energy is less than 1 kcal to form a degenInfo
            degenInfo = degenArray[indDegen,:]
            a,b,c =degenInfo.shape
            ##c=2 always(as it is res, energyVal)#b= number of degen cases#a=1
            degenInfo = degenInfo.reshape(a*b,c)
            #use the index where diff of energy is less than 1 kcal to get degenNames, using the sortedResNamesA
            degenNames = sortedResNamesArr[0:count+1]
            ##Since degenerateNames+denergateArray == DegenInfo, use either!!
            #store info in collectData
            collectData.append([unknownRes, unknownRes, unknownRes.parent.id,'SKIP Less Than ECutoff, degen Info: '+str(degenInfo)+ ' with diff: ' +str(EnDiffWithMin[indDegen])+'<'+str(mc.ECutOff) ])
            skipVal+=1
            #store the struct name
            degenStructNames = get_degenerate_structure_names(degenInfo, debug=debug)
            #update skip info
            skipResInfo.append([unknownRes, degenInfo,degenStructNames,S])
            fLog.write(f"SKIPPING: {structure[modelID][chainID][unknownRes.id[1]]} as lowest vals:{sortedEnArray[0,1]} and {sortedEnArray[1,1]}, degenergate Info: {degenInfo} with {EnDiffWithMin[indDegen]}<{mc.ECutOff}\n")

            fLog.write("###########################################################################\n")
            fLog.flush()
            
            if(debug==1):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f' \n SKIPPING: {structure[modelID][chainID][unknownRes.id[1]]} as lowest vals:{sortedEnArray[0,1]} and {sortedEnArray[1,1]}, degenergate Info: {degenInfo} with {EnDiffWithMin[indDegen]}<{mc.ECutOff} \n', debug=0)


    elif(allEnergyPos or allEnergyNeg):
            ##Degenerate case
            ##In this case all of the values are degenerate but we will double confirm
            MinEn = sortedEnArray[0,1]

            fLog.write(f"ALL ENERGY Values are POS OR Neg: {sortedEnArray}\n")
            fLog.flush()

            #find the difference with minimum energy
            EnDiffWithMin =abs(sortedEnArray[:,1]-MinEn)
            #get the index where difference of energy is less than 1 kcal
            indDegen = np.where(EnDiffWithMin<1)
            #use the index where diff of energy is less than 1 kcal to get degenNames, using the sortedResNamesArr
            degenNames = sortedResNamesArr[indDegen]
            #create the degenArray using the index where diff of energy<1Kcal in the sortedEnArray
            degenArray = sortedEnArray[indDegen]
            #updating the collect data
            collectData.append([unknownRes, unknownRes, chainID,'SKIP Less Than ECutoff, diff: '+str(degenArray)+' with diff: '+str(EnDiffWithMin)+'<'+str(mc.ECutOff) ])


            skipVal+=1
            #getting the degenerate struct names
            degenStructNames = get_degenerate_structure_names(degenArray, debug=debug)
            #updating skipResInfo
            skipResInfo.append([unknownRes, degenArray,degenStructNames,S])

            fLog.write(f"PLS SKP: {structure[modelID][chainID][unknownRes.id[1]]} as lowest vals:{sortedEnArray[0,1]} and {sortedEnArray[1,1]}, degenergate names: {degenNames} and values are: {degenArray} with {EnDiffWithMin}<{mc.ECutOff}\n")
            fLog.write("###########################################################################\n")
            fLog.flush()

            if(debug==1):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"PLS SKP: {structure[modelID][chainID][unknownRes.id[1]]} as lowest vals:{sortedEnArray[0,1]} and {sortedEnArray[1,1]}, degenergate names: {degenNames} and values are: {degenArray} with {EnDiffWithMin}<{mc.ECutOff}\n ###########################################################################\n")
                
    else:
           fLog.write("ERROR: SHOULD NOT GO THROUGH DEGEN CASES. Please go through code!\n")
           fLog.flush()
           if(debug==1):
               stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"ERROR: SHOULD NOT GO THROUGH DEGEN CASES. Please go through code!\n") 
           os.exit()
           
    
    fLog.close()
    return structure, unknownResMod,changeVal, skipVal, collectData, skipResInfo


def evaluate_HIP_cases(unknownRes, structure, S, LOCA_unknownRes, unknownResMod, changeVal, skipVal, collectData, skipResInfo, chV_level, debug =0) :
    '''
    objective: To investigate the HIS could be HIP/HIP rotamer
    I/P:-unknownRes: the unknown HIS to investigate
        -structure: structure to update
        -S: is a dictionary of possible structures corresponsing to HIP/HIE/HID and rotamers
        -LOCA_unknownRes: List of close atoms for the unknown residue:
        -unknownResMod: list of unknown residues that needs to be updated
        -changeVal: change value of the unknown->known residue needs to be tracked as degenerate
                        cases are fixed up
        -skipVal:skip value of the unknown->unknown residue(no change) needs to be tracked as well
        -collectData: collecting data for the changes being done
        -skipResInfo: collecting information where no changes are made-and the unknown residue is skipped.This includes: unknown redisue, degenerate info, degenerate structure names, structure associated
        -chV_level: a string for: changeVal and level- for tracking purposes

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

    '''    

    fLogName = stp.get_log_file_name(debug=0)
    fLog = open(fLogName, "a")          

    modelID = unknownRes.parent.parent.id
    chainID = unknownRes.parent.id

    sHIP = 'struct'+str(unknownRes.id[1])+'HIP' 
    resHIP = S[sHIP][modelID][chainID][unknownRes.id]

    sHIPR = 'struct'+str(unknownRes.id[1])+'HIPR' 
    resHIPR = S[sHIPR][modelID][chainID][unknownRes.id]
    HIPset = 0
    HIPdegen = 0


    enSumTotND1, enSumTotNE2, enSumTotND1Ro, enSumTotNE2Ro   = get_HIP_energies(resHIP, resHIPR, chV_level, debug=debug)    
    sumOrigState = enSumTotND1 + enSumTotNE2
    
    sumRotamer = enSumTotND1Ro + enSumTotNE2Ro
    origStateHIP = enSumTotND1< -1 and enSumTotNE2< -1
    rotamerStateHIP = enSumTotND1Ro< -1 and enSumTotNE2Ro<-1
##################################################################################################################
    #Checking the degenerate case
    if(origStateHIP and rotamerStateHIP and abs(sumOrigState - sumRotamer)<1 and (enSumTotND1 !=0 and enSumTotNE2 !=0 and enSumTotND1Ro !=0 and enSumTotNE2Ro !=0 ) ):
          fLog.write(f"HIP:{unknownRes} of {unknownRes.parent} is DEGENERATE!! \n")
          fLog.flush()
          if(debug==1):
              stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"HIP:{unknownRes} of {unknownRes.parent} is DEGENERATE!! \n", debug=0)
          #create the list and array of energy information!
          energyList = [[resHIP, sumOrigState,S[sHIP], resHIP,'HIP'], [resHIPR, sumRotamer, S[sHIP],resHIPR ,'HIPR']]
          energyArray = np.array(energyList)
          sortedEnList = sorted(energyList,key=lambda x: (x[1]))
          sortedEnArray = np.array(sortedEnList)


          #check the degenerate cases
          skipValb4check = skipVal
          structure, unknownResMod,changeVal, skipVal, collectData, skipResInfo = evaluate_degenerate_cases( unknownRes, structure, S, energyArray, sortedEnArray, unknownResMod, changeVal, skipVal, collectData, skipResInfo, debug=debug)
          if(abs(skipValb4check - skipVal)==0):
              HIPdegen =0
          else:
              HIPdegen = 1
              del S['struct'+str(unknownRes.id[1])+'HIE' ]
              del S['struct'+str(unknownRes.id[1])+'HIER' ]
              del S['struct'+str(unknownRes.id[1])+'HID' ]
              del S['struct'+str(unknownRes.id[1])+'HIDR' ]

          return structure, unknownResMod, changeVal, skipVal, collectData, skipResInfo, S, HIPset, HIPdegen

    elif(enSumTotND1< -1 and enSumTotNE2< -1):
        ##check if the original state is a possibility
        ##Also check if the closest atom is OG from SER or OG1 from THR. If it is then HIP cannot be the state.
        ##Because OG/OG1 are poor acceptors
        fLog.write(f"possibility of original state HIP as the two energy values are: {enSumTotND1} and {enSumTotNE2}\n")
        fLog.flush()
        if(debug==1):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"possibility of original state HIP as the two energy values are: {enSumTotND1} and {enSumTotNE2}", debug=0)
        myHIP = mra.my_residue(resHIP)
        LOAA_unknownRes = myHIP.get_unknown_residue_acceptor_donor_atoms()
        
###############################ADD to check if one of the acceptor pos is SER/THR######################################################################
        ##Get the list of close atoms for the list of active atoms!!
        #This list of close atoms may not contain only known atoms!!
        LOCA_unknownRes, dim = cats.get_list_of_close_atoms_for_list_of_atoms(LOAA_unknownRes,'DONOR_ACCEPTOR_BOTH', debug=debug)
    
        ##dim[0] id number of close atoms for first nitrogen and dim[1] is num of close atoms for second nitrogen.
        ##if there is no interaction with either of the  atoms-then there should NOT be considered a HIP at all since both ineractions must be negetive!!
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
                ##If it cannot be HIP-delete the appropriate structures and move on..
                fLog.write(f"cannotBeHIP: {cannotBeHIP}\n")
                fLog.flush()

                if(debug==1):
                    stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"cannotBeHIP: {cannotBeHIP}\n", debug=0)
                del S['struct'+str(unknownRes.id[1])+'HIP' ]
                del S['struct'+str(unknownRes.id[1])+'HIPR' ]

            else:
                #It is HIP:
                res2keep = resHIP
                nameOfS2keep = sHIP
                saveText = f"Saving residue: {res2keep.resname} to residue state corresponding to minimum energy i.e ND1Ro: {enSumTotND1} + NE2Ro: {enSumTotNE2} isRotamer:{res2keep.isRotamer}"
                structure, unknownResMod, changeVal, collectData = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, collectData, S, saveText, debug=debug)
                HIPset = 1 

                del S['struct'+str(unknownRes.id[1])+'HIE' ]
                del S['struct'+str(unknownRes.id[1])+'HIER' ]
                del S['struct'+str(unknownRes.id[1])+'HID' ]
                del S['struct'+str(unknownRes.id[1])+'HIDR' ]
                del S['struct'+str(unknownRes.id[1])+'HIPR' ]
        
        else: 
            fLog.write("Investigating HIP-where energies for both nitrogen's<-1 but close atoms = 0. PLS CHECK CODE \n")
            fLog.flush()
            
            if(debug==1):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"Investigating HIP-where energies for both nitrogen's<-1 but close atoms = 0. PLS CHECK CODE \n", debug=0)
            os.exit()

    elif(enSumTotND1Ro< -1 and enSumTotNE2Ro<-1):
        ##check if the rotamer state is a possibility
        ##Also check if the closest atom is OG from SER or OG1 from THR. If it is then HIP CANNOT BE THE STATE
        ##Because OG/OG1 are poor acceptors
        fLog.write("possibility of rotomer state HIP...\n")
        fLog.flush()
        if(debug==1):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"possibility of rotomer state HIP...\n", debug=0)
        myHIP = mra.my_residue(resHIP)
        myHIPR = mra.my_residue(resHIPR)

        LOAA_unknownResR = myHIPR.get_unknown_residue_acceptor_donor_atoms()

###############################ADD to check if one of the acceptor pos is SER/THR######################################################################
        #Create the list of close atoms for active active atoms
        #This list of close atoms may not contain only known atoms!!
        LOCA_unknownResR, dim = cats.get_list_of_close_atoms_for_list_of_atoms(LOAA_unknownResR,'DONOR_ACCEPTOR_BOTH', debug=0)
        ##dim[0] id number of close atoms for first nitrogen and dim[1] is num of close atoms for second nitrogen.
        ##if there is no interaction with either of the  atoms-then there should NOT be considered a HIP at all since both ineractions must be negetive!!
        if(dim[0]>1 or dim[1]>1):
            #There are two lists created. One will be for ND1, and other will be for NE2.
            Nlist1 = LOCA_unknownResR[0]
            Nlist2 = LOCA_unknownResR[1]
            
            ##Create an array and a list of close atom names to iterate over and find if one of the close atoms is OG1/OG or SER/THR
            closeAtsArr1 = np.array(Nlist1)[:,0]
            closeAtsArr2 = np.array(Nlist2)[:,0]

            ####The first atom is the reference atom for HIS (ND1/NE2). Both act as a donor in the case of HIP!
            closeAtsArr = [closeAtsArr1, closeAtsArr2]
            
            cannotBeHIPR = []
            
            ##Iterate over the two close atom array, and over each atom in that array to look for OG/OG1.
            #If found append to cannot be HIP
            for cAarray in closeAtsArr:
                for atom in cAarray:
                    if((atom.name=='OG' and atom.parent.resname == 'SER') or (atom.name=='OG1' and atom.parent.resname == 'THR')): 
                        cannotBeHIPR.append(1)

            if(any(value == 1 for value in cannotBeHIPR)):
                ##If it cannot by HIP-delete the appropriate structures and move on..
                fLog.write(f"cannotBeHIPR: {cannotBeHIPR} \n")
                fLog.flush()

                if(debug==1):
                    stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"cannotBeHIPR: {cannotBeHIPR} \n", debug=0)
                del S['struct'+str(unknownRes.id[1])+'HIP' ]
                del S['struct'+str(unknownRes.id[1])+'HIPR' ]

            else: 
                #It is HIP rotamer:
                res2keep = resHIPR
                nameOfS2keep = sHIPR
                saveText = f"Saving residue: {res2keep.resname} to residue state corresponding to minimum energy i.e ND1Ro: {enSumTotND1Ro} + NE2Ro: {enSumTotNE2Ro} isRotamer:{res2keep.isRotamer}"
                structure, unknownResMod, changeVal, collectData = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, collectData, S, saveText, debug=debug)
                HIPset = 1 

                del S['struct'+str(unknownRes.id[1])+'HIE' ]
                del S['struct'+str(unknownRes.id[1])+'HIER' ]
                del S['struct'+str(unknownRes.id[1])+'HID' ]
                del S['struct'+str(unknownRes.id[1])+'HIDR' ]
                del S['struct'+str(unknownRes.id[1])+'HIP' ]

        else: 
            fLog.write("Investigating HIP Rotamer-where energies for both nitrogen's<-1 but close atoms = 0. PLS CHECK CODE \n")
            fLog.flush()
            if(debug==1):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Investigating HIP Rotamer-where energies for both nitrogens<-1 but close atoms = 0. PLS CHECK CODE \n", debug=0)
            os.exit()

    else:
        del S['struct'+str(unknownRes.id[1])+'HIP' ]
        del S['struct'+str(unknownRes.id[1])+'HIPR' ]

    fLog.close() 
    return structure, unknownResMod, changeVal, skipVal, collectData, skipResInfo, S, HIPset, HIPdegen


def iterate_list_of_unknown_residues_and_set_states(structure,level, chV_level, numCount=1, debug=0):
    ''' 
    objective: To iterate over a given structure and to set states depending on energy computations
    I/P: structure: structure of a protein
         level: the vertical level we are at
         chV_level: a string that combines: change value in a given structure and level value. Used for debug/tracking purposes.
         numCount: structure number on this level

    O/P: structure: updated structure
         changeVal: number of changes that occur
         skipVal: num of residues skipped
         skipResInfo: collecting information where no changes are made-and the unknown residue is skipped. This includes: unknown redisue, degenerate info, degenerate structure names, structure associated
        '''
    fLogName = stp.get_log_file_name(debug=0)
    fLog = open(fLogName, "a")

    outputFolder = stp.get_output_folder_name(debug=0)

    skipVal = 0
    changeVal =0 

    collectData = []
    skipResInfo = []
    ##We need two copies-as one is the list we will go over and the other is the list that will be updated!!
    unknownResIter = stp.get_unknown_residue_list(structure, debug=0)
    unknownResMod = copy.deepcopy(unknownResIter)

    lenUnResOrig = len(unknownResIter)

    unknownResNames = []
    for res in unknownResIter:
        unknownResNames.append(res.resname)

    uniqRes = Counter(unknownResNames).items()
    fLog.write(f"\n Level:{level}, structure number on this level: {numCount}, unknown residue to iterate over: {unknownResIter}, and its length: {len(unknownResIter)}\n\n")
    fLog.write(f"\n The unknowns present: {uniqRes} \n\n")
    fLog.flush()

    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"\n Level:{level}, structure number on this level: {numCount}, unknown residue to iterate over: {unknownResIter}, and its length: {len(unknownResIter)}\n\n The unknowns present: {uniqRes} \n\n", debug=0)
    
    
    for count, unknownResOrig in enumerate(unknownResIter):    
        modelID = unknownResOrig.parent.parent.id
        chainID = unknownResOrig.parent.id
        unknownRes = structure[modelID][chainID][unknownResOrig.id]
        if(unknownRes.isKnown == 1):continue

        fLog.write("\n###############################################################################\n")
        fLog.write("###############################################################################\n\n")
        fLog.write(f"unknown residue number: {count+1}/{lenUnResOrig} and unknown res is: {unknownRes} and its known val:{unknownRes.isKnown} and is rotamer:{unknownRes.isRotamer}, SKPVAL:{skipVal}, ChangeVal: {changeVal}\n")
        fLog.flush()
        
        if(debug==1):
            stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f" ###############################################################################\n ###############################################################################\n\n Number: {count+1}/{lenUnResOrig} and unknown res is: {unknownRes} and its known val:{unknownRes.isKnown} and is rotamer:{unknownRes.isRotamer}, Skip value:{skipVal}, ChangeVal: {changeVal}\n", debug=0)
    
        myUnknownRes = mra.my_residue(unknownRes)
        #Get list of active atoms, list of close atoms, and number of close atoms
        LOAA_unknownRes = myUnknownRes.get_unknown_residue_acceptor_donor_atoms()
        ##This adds close points for rotomer HIS as well:
        if(unknownRes.resname == 'HIS'):
            LOAA_unknownRes.append(structure[modelID][chainID][unknownRes.id]['CD2'])
            LOAA_unknownRes.append(structure[modelID][chainID][unknownRes.id]['CE1'])
    
        #Get all the close atoms for the atoms in concern
        #This list of close atoms may not contain only known atoms!!
        LOCA_unknownRes, dim = cats.get_list_of_close_atoms_for_list_of_atoms(LOAA_unknownRes,'DONOR_ACCEPTOR_BOTH_TBD', debug=debug)
        ###Check if close atoms are present. If not, HIS->HIE, ASN/GLN remain the same. Now these are known!
        if(all(x < 2 for x in dim)):
            fLog.write(f"NO CLOSE ATOMS FOUND!\n\n")
            fLog.flush()
            if(debug==1):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"NO CLOSE ATOMS FOUND!\n\n", debug=0)
            
            if(unknownRes.resname=='HIS'):  dictStructHIE = setup_HIS(unknownRes, structure, 'HIE', debug=debug)

            collectData.append([unknownRes, dictStructHIE[modelID][chainID][unknownRes.id],chainID,'DONE: NO CLOSE ATOMS FOUND'])

            structure[modelID][chainID][unknownRes.id].isKnown = 1 
            unknownRes.isKnown = 1
            changeVal +=1
            unknownResMod.remove(unknownRes)
        else:
            fLog.write(f"checked for close atoms and close atoms are present!!\n\n")
            fLog.flush()
            if(debug==1):
                stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"checked for close atoms and close atoms are present!!\n\n", debug=0)


            flatLOCA = [item for sublist in LOCA_unknownRes for item in sublist]
            flatLOCAarray = np.array(flatLOCA)
            
            S = branch_structure(unknownRes, structure, debug=debug)
            ##Note: we do not convert it into known at this point as it is only done when setting a state. Here we just create branch to analyze the different rotameric/protonation states.
            #create energy List/array, residue states, sorted energy list/array
            energyList, resStates = compute_energy_for_all_states(unknownRes, S, chV_level, debug=debug)

            nameOfSposs = mc.unResDict[unknownRes.resname][0]
            energyArray = np.array(energyList, dtype=object)
            sortedEnList = sorted(energyList,key=lambda x: (x[1]))
            sortedEnArray = np.array(sortedEnList, dtype=object)

            allEnergyZero = all(c == 0 for c in energyArray[:,1])    ##All close atoms have zero energy

            if(unknownRes.resname == 'HIS' and not(allEnergyZero)): 
                #If HIS and all energy values are not zero-test the possibility of HIP:
                structure, unknownResMod, changeVal, skipVal, collectData, skipResInfo, S, HIPset, HIPdegen = evaluate_HIP_cases(unknownRes, structure, S, LOCA_unknownRes, unknownResMod, changeVal, skipVal, collectData, skipResInfo, chV_level, debug=debug)

                if(HIPdegen==1 or HIPset==1):
                    fLog.write("Continuing to the next unknown in the for loop as HIP is set or HIP is degenerate\n")
                    fLog.flush()

                    if(debug==1):
                        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Continuing to the next unknown in the for loop as HIP is set or HIP is degenerate\n", debug=0)
                    continue ##continue to next value in the for loop

                fLog.write("continuing as HIP is not an option!!\n") ##continue in the for loop
                fLog.flush()
                if(debug==1):
                    stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"continuing as HIP is not an option!!\n", debug=0)

            else:
                #If not a HIS/all HIS energy values are zero. Declare we are not testing for HIP!
                fLog.write(f"Not checking for HIP- as I am {unknownRes} of {unknownRes.parent} and I am not on HIS or all my HIS energies are ZERO\n\n")
                fLog.flush()
                if(debug==1):
                    stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f"Not checking for HIP- as I am {unknownRes} of {unknownRes.parent} and I am not on HIS or all my HIS energies are ZERO\n\n", debug=0)
    #########Take the smallest value and subtract from all other values. Find out how many values are<1
            MinEn = sortedEnArray[0,1]
            #The first value is the minimum. So starting from second Val
            diffWithMin = abs(sortedEnArray[1:,1]-MinEn)
            MinTwoEnDiff = abs(sortedEnList[0][1]-sortedEnList[1][1]) 
            ##checking if the difference between the smallest two energy value is less than 1-thus a possible degeneracy!
            degenFound = any(diffWithMin< mc.ECutOff) 
###################################IF DEGENERATE ###############################################################
            if(degenFound):
                #if smallest two energies are less than ECutOff then explore the possibility of degeneracy 
                structure, unknownResMod,changeVal, skipVal, collectData, skipResInfo = evaluate_degenerate_cases(unknownRes, structure, S, energyArray, sortedEnArray, unknownResMod, changeVal, skipVal, collectData, skipResInfo, debug=debug) 
            else:
                ##IF NOT DEGENERATE, pick the state with the smallest energy value:
                res2keep = sortedEnArray[0,0] ##Picking the smallest as 0<+v, -v<0
                ind = 0
                moreInfoName = sortedEnArray[0,-1]
                nameOfS2keep = get_structure_name(res2keep, moreInfo = moreInfoName, debug=debug)

                enValPick =sortedEnArray[ind][1]
                saveText = f"State Set! The energies of :{sortedEnArray[0,0]} is {sortedEnArray[0,1]} and {sortedEnArray[0,0]} is {sortedEnArray[1,1]}, Saving residue: {res2keep.resname} to residue state corresponding to minimum energy  i.e {enValPick} and isRotamer:{res2keep.isRotamer}"

                structure, unknownResMod, changeVal, collectData = set_state(unknownRes, res2keep, nameOfS2keep, unknownResMod, changeVal, collectData, S, saveText, debug=debug)
                del S
    fLog.close()
   
    return structure,changeVal,skipVal,skipResInfo


def resolve_residue_ambiguities_in_structure(structure,fOutName = 'out', maxLevel=5, chValMax=5,debug=0): 
    '''
    objective: resolve ambiguity of the unknown residues in a given structure
    I/P: -structure: structure of the protien(it is a class data structure)
         -maxLevel: max number of levels we want to traverse
         -chValMax: Max number of change value in each level to accomodate

    O/P: -pdbFileNum: number of PDB files created
         -listOfStructsAll: list of all structs created during the traversal
         -skipInfoAll: Information of all residues skipped in each structure formed
         -skipValAll: skip values during each struct formation
    '''
    

    listOfStructsCurr=[structure]
    listOfStructsAll=[structure]
    skipInfoAll=[]
    skipValAll=[]
    filesGen=[]

    level=0
    pdbFileNum=0
 
    outputFolder = stp.get_output_folder_name(debug=debug)
    pdbOutFolder = stp.get_pdb_out_folder(debug=debug)

    fLogName = stp.get_log_file_name(debug=0)
    fLog = open(fLogName, "a")

    for level in range(maxLevel):
        level+=1
        fLog.write(f"Starting a new level:{level}\n")
        fLog.flush()
        
        if(debug==1):
            stp.append_to_debug( __name__, sys._getframe().f_code.co_name,f"Starting a new level:{level}\n", debug=0)

        #current level S Len is the number of structures in a given level. Horizontal traversal first and then vertical!
        #This is number of structures in horizontal traversal!!
        currLevelSLen = len(listOfStructsCurr)

        if(currLevelSLen == 0):
            #if there are no structures in the current level then break out as all the files to be written are also done.
            fLog.write("Current S length =0! No more structs to explore\n") 
            fLog.flush()
            if(debug==1):
                stp.append_to_debug( __name__, sys._getframe().f_code.co_name,"Current structure list(S) length =0! No more structs to explore\n", debug=0) 

            break
        if(level > maxLevel-1): 
            fLog.write("ERROR LEVEL: Exceeding max level limit!\n")
            fLog.flush()
            if(debug==1):
                stp.append_to_debug( __name__, sys._getframe().f_code.co_name,"ERROR LEVEL: Exceeding max level limit!\n", debug=0)
        
        #to store the newly created list of structures for the next level
        LOS_newLevel = []
        LOS_newLevel_dicts = []
        fLog.write(f"In the OUTER MOST loop where level is: {level} and length of structure list(S):{currLevelSLen}\n")
        fLog.flush()
        if(debug==1):
            stp.append_to_debug( __name__, sys._getframe().f_code.co_name,f"In the OUTER MOST loop where level is: {level} and length of structure list(S):{currLevelSLen}\n", debug=0)

        #Need to iterate over each structure in the current level:
        for st in range(currLevelSLen):
            chVal = 1
            count = 0 
            structCurr = listOfStructsCurr[st]
            
            #Going over max amount of change Values that can occur
            for chV in range(chValMax):
                chV+=1
                #if exceeding max-human intervention is required!
                if(chV > chValMax-1): 
                    fLog.write("\n##########################################\n")
                    fLog.write("\nERROR CHV: Exceeding max chV level limit!\n")
                    fLog.write("\n##########################################\n")
                    fLog.flush()
                    if(debug==1):
                        stp.append_to_debug( __name__, sys._getframe().f_code.co_name,\
                                        "\n##########################################\n \
                                         \nERROR CHV: Exceeding max chV level limit!\n \
                                         \n##########################################\n", debug=0)



                    
                if(chVal>0):
                    #as long as change value is positive, keep iterating over list of unknown residues
                    count +=1
                    chV_level = "level_"+str(level)+"_chV_" + str(chV) +"_structureNum_"+str(st)
                    fLog.write("\n###########################################################\n #############################################################\n")
                    fLog.write(f"chV(in range of chValMax or max amount of change Values that can occur ={chValMax}): {chV}, chVal(or number of residue changed in this level):{chVal} current level: {level}, st:{st}/{currLevelSLen-1}, counting number of times a change occurs: {count-1}\n")
                    fLog.flush()

                    if(debug==1):
                        stp.append_to_debug( __name__, sys._getframe().f_code.co_name,\
f"\n###########################################################\n #############################################################\n chV(in range of chValMax or max amount of change Values that can occur ={chValMax}): {chV}, chVal(or number of residue changed in this level):{chVal} current level: {level}, st:{st}/{currLevelSLen-1}, counting number of times a change occurs: {count-1}\n", debug=0)
 

                    #iterate over all the unknown residues and set as many states as possible. If a change is made or a state is set then chVal increments by 1 else skpVal increments by 1
                    structNew, chVal, skpVal, skpInfo = iterate_list_of_unknown_residues_and_set_states(structCurr,level, chV_level, numCount=count, debug=debug)
                    skipInfoAll.append(skpInfo)
                    skipValAll.append(skpVal)
                    fLog.write(f"chVal after goin through the iterative loop: {chVal}\n")
                    fLog.flush()

                    if(debug==1):
                        stp.append_to_debug( __name__, sys._getframe().f_code.co_name,\
                                f"chVal after goin through the iterative loop: {chVal}\n", debug=0)


                    if(chVal == 0):
                        #If change value = 0, then compute list of unknown residues else update structCurr to struct new and repeat.
                        LOURch0 = stp.get_unknown_residue_list(structNew, debug=0)
                        #If change value  =0, and there are still unknowns in the list-it is time to branch!!
                        if(LOURch0):

                           fLog.write(f"level:{level},In resolve ambiguity, in List Of Unknown Reside change0 (LOURcho): unknown {LOURch0} and length: {len(LOURch0)}\n")

                           fLog.flush()
                           if(debug==1):
                               stp.append_to_debug( __name__, sys._getframe().f_code.co_name,\
                                f"chVal after goin through the iterative loop: {chVal}\n", debug=0)


                    if(chVal == 0):
                        #If change value = 0, then compute list of unknown residues else update structCurr to struct new and repeat.
                        LOURch0 = stp.get_unknown_residue_list(structNew, debug=debug)
                        #If change value  =0, and there are still unknowns in the list-it is time to branch!!
                        if(LOURch0):

                           fLog.write(f"level:{level},In resolve ambiguity, in List Of Unknown Reside change0 (LOURcho): unknown {LOURch0} and length: {len(LOURch0)}\n")

                           fLog.flush()
                           if(debug==1):    
                               stp.append_to_debug( __name__, sys._getframe().f_code.co_name,\
                                       f"level:{level},In resolve ambiguity, in List Of Unknown Reside change0 (LOURcho): unknown {LOURch0} and length: {len(LOURch0)}\n", debug=0)
                           #Create the branch:
                           #Get the name of struct of for the degen cases 
                           namesOfStructs = skpInfo[0][-2]
                           S2branch = {}
                           for name in namesOfStructs:
                                modelID = LOURch0[0].parent.parent.id
                                chainID = LOURch0[0].parent.id

                                structBranch = skpInfo[0][-1][name]
                                structBranch[modelID][chainID][LOURch0[0].id].isKnown = 1
                                if(LOURch0[0].resname == 'ASP' or LOURch0[0].resname == 'GLU'):
                                    sUn = LOURch0[0].parent.parent.parent
                                    #code.interact(local = locals())
                                    unknownASPpair = get_ASP_GLU_pair(LOURch0[0], sUn, mc.deltaD, debug=debug)
                                    modelID_ASPpair = unknownASPpair.parent.parent.id
                                    chainID_ASPpair = unknownASPpair.parent.id
                                    structBranch[modelID_ASPpair][chainID_ASPpair][unknownASPpair.id].isKnown=1
                                name2ref = name +'_level_' +str(level)
                                S2branch[name2ref] = structBranch

                           ##Make the first residue "KNOWN" as we are taking all possible states
                           #save as dictionary for easy access while inspection
                           LOS_newLevel_dicts.append(S2branch)
                           LOS_newLevel.append(list(S2branch.values()))
                           listOfStructsAll.append(S2branch)
                           fLog.write(f"printing all list of structs: {listOfStructsAll}, its length: {len(listOfStructsAll)}\n")
                           fLog.write(f"I am breaking out of loop as ChVal = {chVal} and branch of S has been created: {S2branch}\n")
                           fLog.flush()

                           if(debug==1):
                               stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"printing all list of structs: {listOfStructsAll}, its length: {len(listOfStructsAll)}\n I am breaking out of loop as ChVal = {chVal} and branch of S has been created: {S2branch}\n", debug=0)
                           break###Breaking out of chValMax for loop
                        else:
                            ###If there are no more unknown residue, pls WRITE2PDB
                            fPDBfullPath = os.getcwd()+f"/{pdbOutFolder}/{fOutName}_{pdbFileNum}.pdb"
                            fPDBname = f"{fOutName}_{pdbFileNum}.pdb"
    
                            fLog.write(f"I am now writing a file: {fPDBname} since currently my state is: chV: {chV}, chVal:{chVal} current level: {level}, st:{st}/{currLevelSLen-1}, counting number of times a change occurs: {count}\n")  
                            fLog.flush()
                            
                            if(debug==1):
                                stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"I am now writing a file: {fPDBname} since currently my state is: chV: {chV}, chVal:{chVal} current level: {level}, st:{st}/{currLevelSLen-1}, counting number of times a change occurs: {count}\n", debug=0)
        
                            stp.detect_clash_within_structure(structNew,withinStructClashDist = mc.btwResClashDist, debug=debug)
                            stp.detect_clash_within_residue_for_all_residues(structNew, withinResClashDist =mc.withinResClashDist, debug=debug)
        
                            filesGen.append(fPDBfullPath)
                            #keep a track on number of pdb files written
                            pdbFileNum+=1  
                            stp.write_to_PDB(structNew, fPDBfullPath, removeHLP=True, debug=debug) 
                            
                            break
                    else:
                        #If change value is not yet zero-keep iterating over by updating the current struct to new struct
                        fLog.write("\nNow assigning the new struct to current struct\n")
                        fLog.flush() 

                        if(debug==1):
                            stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"\n Now assigning the new struct to current struct\n", debug=0)
                        structCurr = structNew
                else:
                     #If change value is not greater than 0 then break out of the loop.
                     fLog.write(f"change value is not greater than 0. I am breaking out. \n")
                     fLog.flush()
                     if(debug==1):
                        stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"change value is not greater than 0. I am breaking out. \n", debug=0)
                     break

        #flatten the list of structures that are created for the new iteration and update list of current strucutres
        listOfStructsCurr = [item for sublist in LOS_newLevel for item in sublist]
    fLog.close()
    return pdbFileNum, filesGen,listOfStructsAll,skipInfoAll, skipValAll

