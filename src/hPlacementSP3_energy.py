import os
import sys
import Bio
import numpy as np
from Bio.PDB import *

import code
import math

import myResAtom as mra
import myConstants as mc
import setupProt as stp
import hPlacementSP2 as hsp2
import closeAtoms as cats
import myMath as mm

def checkBondAngForAllAAt(sp3,abvSP3, sp3Coord, abvSP3Coord, aat0, aat1, aat2, debug=0):

    ''' Objective: checks/prints the bond angle with all attached atoms
        Input: -sp3: sp3 atom
              -abvSP3: atom connected with sp3 that is not hydrogen or LP
              -sp3Coord: sp3 atom coord
              -abvSP3Coord: atom connected with sp3 that is not hydrogen or LP coord

              -aat0, aat1, aat2: Hyd/LP atoms
              
        Output: Prints out the angle between above SP3 atom-SP3 atom-hyd/LP atom

    '''
    sp3Vec = Vector(sp3Coord)
    abvSP3Vec = Vector(abvSP3Coord)
    aat0Vec =Vector(aat0)
    aat1Vec =Vector(aat1)
    aat2Vec =Vector(aat2)
    stp.append2debug( __name__, sys._getframe().f_code.co_name,\
            f" Checking for sp3: {sp3} of residue: {sp3.parent} of chain: {sp3.parent.parent}\n \
               aboveSP3 is: {abvSP3} of residue: {sp3.parent} of chain: {sp3.parent.parent}\n \
               sp3 angle should be approximately 109.5 degrees.\n \
               Calc angle:aboveSP3-SP3-Hcoord0 {calc_angle(abvSP3Vec, sp3Vec, aat0Vec)*180/3.14}, \n \
               Calc angle:aboveSP3-SP3-Hcoord1/LP1 {calc_angle(abvSP3Vec, sp3Vec, aat1Vec)*180/3.14},\n \
               Calc angle:aboveSP3-SP3-Hcoord2/LP2 {calc_angle(abvSP3Vec, sp3Vec, aat2Vec)*180/3.14}, \n \
    ", debug=0)


def computeInitialSp3BondedAtomCoord(sp3Coord, surrCoord, bondLength, debug =0 ):
   
    ''' 
    Objective: Calculate the initially attached atom/lone pair coordinate value. This
    may not be the actual value as we will be transforming into another frame and
    rotating to find minimum distance sum btw acceptors &  h-donors. 
    Note-Adjustment to the angle of 109.5 is done at a late point

    Input:
         sp3Coord: coordinate of sp3 atom
         surrCoord: the surrounding atom or the closest atom wrt sp3 atom
         bondLength: bond length between sp3 atom and hydrogen/LP atom = 1 A

    Output:
         aat0PosVec: compute position vector of the first atom (Hyd/LP) connected with sp3 atom.
                    Note-Adjustment to the angle of 109.5 is done at a late point

    '''
    #Find the direction
    sp3AAt0 = bondLength*mm.unitVec(surrCoord-sp3Coord, debug=0) 
    #Find the position vector
    aat0PosVec = sp3AAt0 + sp3Coord

    return aat0PosVec


def createPrimeFrameOfReference(sp3Coord, aboveSp3Coord, aat0Coord, debug =0): 
    '''   
        Objective: 
        Creating the new rotated frame of reference (the unit vectors) 
        assuming the vector from sp3 atom to the above_sp3_coord as the 
        zHatPrime, and to find the xHatPrime: use the vector from sp3 to 
        h0 and removing its zHatPrime component to get a vector in
        xHatPrime and yHatPrime. We will later translate the frame of
        ref s.t yHatPrime =0,so:
        xHatPrime = sp3_h0-(sp3 dot zHatPrime)zHatPrime   
        yHatPrime = crossProd( zHatPrime, xHatPrime)


        I/P: sp3_coord, above_sp3_coord, h0_coord
        O/P: xHatPrime, yHatPrime, zHatPrime

        '''


    zHatPrime = mm.unitVec(aboveSp3Coord - sp3Coord, debug=debug)
    ##Find x_hat_prime in original frame
    ###Remove the z_hat_prime component of nitrogen_hydrogen vector/r1 from r1 to get
    ##x_hat_prime and y_hat_prime component. But by def, we are taking y_hat_prime = 0. 
    ##So we find the vector in x_hat_prime component.
    ##Another way to think:: r2 = r1 -r3, and r3 = (r1 dot z_hat_prime)*z_hat_prime.
    ##BD/r1 Vector:
    sp3AAt0 =(aat0Coord - sp3Coord)
    #Create new frame of reference using above information.
    sp3AAt0InZhatPrime = np.dot(sp3AAt0, zHatPrime)*zHatPrime
    xPrime = sp3AAt0 - sp3AAt0InZhatPrime

    xHatPrime = mm.unitVec(xPrime, debug=debug)
    yHatPrime = np.cross(zHatPrime, xHatPrime)

    if(debug==1):
        stp.append2debug( __name__, sys._getframe().f_code.co_name, f"New frame of reference has unit vectors: {xHatPrime}, {yHatPrime}, {zHatPrime}!", debug=0)


    return xHatPrime, yHatPrime, zHatPrime


def setupTransformMatrix( xHatPrime, yHatPrime, zHatPrime, debug=0):
    
    '''Set up the transform matrix '''

    return np.array([xHatPrime, yHatPrime, zHatPrime])


def moveCoordsToPrimeFrameOfReference(shiftByCoord, transformMatrix, allCoordsToMove, debug=0):

    ''' objective: Move the desired coordinates to the prime(new) frame of reference from original frame of reference
        Input: 
            -shiftByCoord: shift or translate by coords 
            -transformMatrix: Transformation matrix for rotation(x prime, y prime, z prime)
            -allCoordsToMove: The coords to translate and rotate

        Output: transformedCoords: Newly transformed( translated/rotated) coordinates
    '''
 
    ##First translate then rotate!!
    translatedCoords = mm.translateCoord(shiftByCoord, allCoordsToMove, debug=debug)
    ###Now doing a matmult or einsum to get transformedCoords = np.matmul(transformMatrix, translatedCoords)
    if(len(translatedCoords.shape)== 1):
        transformedCoords = np.einsum('ij,kj->ki', transformMatrix, [translatedCoords] )
    else:
        transformedCoords = np.einsum('ij,kj->ki', transformMatrix, translatedCoords ) 

    return transformedCoords



def moveCoordsToOriginalFrameOfReference(shiftCoord, transformMatrix, moveCoordsBack, debug=0):

    ''' Objective: Move the desired coordinates to the original frame of reference from new frame of reference 
        Input: -shiftCoord/sp3Coord: Coordinates to translate or shift by (coordinate of sp3 atoms)
               -transformMatrix: Transform or Rotation matrix to get back to original frame of ref(xprime,yprime,zprime)
               -moveCoordsBack: Coords to move
        Output:-hBack: original/Transformed to original coordinate of the moveCoordsBack 

    '''
    hBack = np.einsum("ij,kj->ki", np.linalg.inv(transformMatrix), moveCoordsBack) + shiftCoord    
    return hBack


def getAllAttachedAtoms(aat0CoordPrime, debug=0): 

    ''' 
        Objective: Get the other two attached atom coords given one attached atom coord in the new frame by rotating   
        Input: -aat0CoordPrime: coordinate of the first attached atom of the sp3 atom
        Output: -aatPrime: array of the three attached atom to the sp3 atom that is 109.5 degrees apart
    '''
    #Three atoms in plane at 120 degrees each!
    rotInPlane = np.array((mm.rotMatrix(120, debug=debug),mm.rotMatrix(240, debug=debug)))

    aat1CoordPrime, aat2CoordPrime = np.einsum('ijk,k->ij',rotInPlane, aat0CoordPrime)
    aatPrime = np.array((aat0CoordPrime, aat1CoordPrime, aat2CoordPrime))
    
    return aatPrime 

    
def energyValOfACATSforHLP(sp3, aboveSp3, aatPrime, allCloseAtoms, ang, debug=0):

    '''
        objective: energy value of all close Atoms for Hydrogens and Lone Pairs
                    To find energy value of all close atoms for Hyd-0/Hyd-1/Hyd-2  is Sp3.parent ==LYS
                    or  Hyd-0/LP-1/LP-2 if sp3.parent == SER/THR

        input:-sp3: Sp3  atom
              -aboveSp3: atom connected to sp3 which is not Hydrogen/lone pair
              -aatPrime: active atom prime coordinates (Hyd/lone pair)
              -allCloseAtoms: All close atoms
              -ang: angle at which energy values are being computed

        output: -aat0Energy, aat1Energy, aat2Energy: energies wrt  Hyd-0/Hyd-1/Hyd-2 or Hyd-0/LP-1/LP-
                -enSum: sum of the energies 

    '''
    ##energy summation
    enSum = 0
    ##Additional information for debug purpose:ang, sp3.parent.resname, sp3.parent.id[1], closeAt.parent.resname, closeAt.parent.id[1], aat0Energy, aat1Energy, aat2Energy, enSum

    energyInteraction = []
    energyInteraction.append(['angle', 'sp3', 'sp3.parent', 'closeAt', 'closeAt.parent', 'aat0Energy, aat1Energy', 'aat2Energy', 'enSum'])
    ####################################################################################################
    for i in range(1, np.shape(allCloseAtoms)[0]):
        ##close atoms for which energy needs to be computed 
        closeAt = allCloseAtoms[i][0]
        myCloseAt = mra.myAtom(allCloseAtoms[i][0])
        enValDon = 0
        enValAcc = 0
        behav = myCloseAt.get_behavior().abbrev
        ##check if close atom is a donor or acceptor (it cant be both)Close atoms to begin with are only known donor or known acceptor. Can we have known "both"? That may improve results??

        if(myCloseAt.get_behavior().abbrev == 'ac'):
            ##Assume the first to be the Hydrogen atom (the others can be either two hydrogens-LYS or two lone pairs-SER/THR)
            aat0Energy, aat0enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[0], sp3, attractive=1, atype = 'SP3',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)

            if(sp3.parent.resname == 'LYS'):
                #For the case of two hydrogens:
                aat1Energy, aat1enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[1], sp3,  attractive=1, atype = 'SP3',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)
                aat2Energy, aat2enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[2], sp3, attractive=1, atype = 'SP3',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)

            else:
                aat1Energy, aat1enSum = cats.computeEnergyAsAcceptor(sp3, Vector(aatPrime[1]), closeAt, attractive = 0, atype = 'SP3',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)
                aat2Energy, aat2enSum = cats.computeEnergyAsAcceptor(sp3, Vector(aatPrime[2]), closeAt, attractive = 0, atype = 'SP3',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)
                
            enSum = enSum + aat0enSum + aat1enSum + aat2enSum


        if(myCloseAt.get_behavior().abbrev == 'do'):
            ##Assume the first to be the Hydrogen atom (the others can be either two hydrogens-LYS or two lone pairs-SER/THR)
            aat0Energy, aat0enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[0], sp3, attractive=0, atype= 'SP3',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)

            if(sp3.parent.resname == 'LYS'):
                #For the case of two hydrogens:
                aat1Energy, aat1enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[1], sp3, attractive =0, atype = 'SP3',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)
                aat2Energy, aat2enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[2], sp3, attractive =0, atype = 'SP3',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)

            else:
                aat1Energy, aat1enSum = cats.computeEnergyAsAcceptor(sp3, Vector(aatPrime[1]), closeAt, attractive = 1, atype = 'SP3',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)
                aat2Energy, aat2enSum = cats.computeEnergyAsAcceptor(sp3, Vector(aatPrime[2]), closeAt, attractive = 1, atype = 'SP3',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)

            enSum = enSum + aat0enSum + aat1enSum + aat2enSum

        energyInteraction.append([ang, sp3, sp3.parent, closeAt, closeAt.parent, aat0Energy,aat1Energy, aat2Energy, enSum])
    
    if(debug == 1):
        stp.append2debug( __name__, sys._getframe().f_code.co_name,f"interaction energies for each close atom with respect to Hydrogen, Hydrogen, Hydrogen incase of LYS and Hydrogen, lone pair, lone pair incase of THR/SER. \
    \n {energyInteraction} ", debug=0) 

    return aat0Energy, aat1Energy, aat2Energy, enSum
#

def optimizeByRotationInPlane_energy(sp3, aboveSp3, aatPrime,allCloseAtoms, debug =0 ):
    '''
        objective: Rotate in plane the three attached atom coordinates and 
         find sum of h-bonding energies  
        
        input:-sp3:sp3 atom
              -aboveSp3:atom connected to sp3 (which is not hydrogen or lone pair)
              -aatPrime:prime coordinate of the attached active atoms
              -allCloseAtoms: all close atoms
        
        output: -S: array of angle and energy at that angle
                -aatPrimeMat: attached active atom prime coordinates at each angle 

    '''
    ####Minimize distance by changing h0_prime, h1_prime, h2_prime####################
    sumEnergyInfo = [] 
    aatPrimeMat = []
    aatSurrAllAngles = []
    #end angle for LYS: 120, and end angle for SER/THR: 360
    endAng = mc.optRotAngle_LYS if(sp3.parent.resname == 'LYS') else mc.optRotAngle
    
    for ind,ang in enumerate(range(0,endAng,10)):
        aatPrimeRot = np.einsum("ij,kj->ki", mm.rotMatrix(ang, debug=debug), aatPrime)
        if(debug==1):
            stp.append2debug( __name__, sys._getframe().f_code.co_name, f"\n From the module: {__name__} and function:{sys._getframe().f_code.co_name}\n Currently at angle: {ang} and now will check all bond angles for: {sp3} of {sp3.parent} of {sp3.parent.parent}", debug=0)
            checkBondAngForAllAAt(sp3, aboveSp3, sp3.coord, aboveSp3.coord, aatPrimeRot[0], aatPrimeRot[1], aatPrimeRot[2],debug=debug )
        aatPrimeMat.append(aatPrimeRot)   

        ##aat1 and aat2 will be acceptors if SER/THR or aa1 and aat2 will be donors if LYS #ACATS = All Close Atoms
        aat0Energy, aat1Energy, aat2Energy, enSum = energyValOfACATSforHLP(sp3, aboveSp3, aatPrimeMat[ind], allCloseAtoms, ang, debug=debug)               
        sumEnergyInfo.append([ang, enSum])

    #outputting it as an array
    S = np.array(sumEnergyInfo)
    if(debug == 1):
        stp.append2debug( __name__, sys._getframe().f_code.co_name, \
                f"\n For atom: {sp3} of {sp3.parent} of {sp3.parent}, side chain hydrogen is needed. \
                \n We search in the plane at an interval of 10 degrees for minimum energy.\
                \n Angle(degrees) | energy value at the angle\n \
                sum energy info: {S}", debug=0)
    
    return S, aatPrimeMat

def computeAAtInNewFrameOfReference_energy(sp3, aboveSp3, allCloseAtoms, aat0CoordPrime, debug=0):
    ''' 
        objective: compute optimized( or minimum energy interaction) coordinate of the attached atoms(H0/H1/H2 or H0/LP1/LP2) in the new frame of reference using the energy method
        input:  -sp3: sp3 atom
                -aboveSp3: atom connected to the sp3 atom(not hydrogen or lone pairs)
                -allCloseAtoms: all relevant(knowns-donor/acceptor) closeAtoms that need to be explored
                -aat0CoordPrime: coordinate of first attached active atom in the prime frame of reference

        output: -aatCoordPrime: optimized coordinate value of the attached atoms
                -S: array of angle and energy at that angle
                -aatPrimePositionMat: all coordinate possibilities(rotated in the plane) of attached atom 

         
    '''

    #get all the attached atom
    aatPrime = getAllAttachedAtoms(aat0CoordPrime, debug=debug) 


    if(debug==1):
        checkBondAngForAllAAt(sp3, aboveSp3, sp3.coord, aboveSp3.coord, aatPrime[0], aatPrime[1], aatPrime[2],debug=debug)
    #############################################################################
    #Find the coordinates for hydrogen and lone pair placement by rotating in plane and computing S i.e angle and energy computed at that angle, along with aatPrimePositionMat or attached atom coordinates
    S, aatPrimePositionMat  = optimizeByRotationInPlane_energy(sp3, aboveSp3, aatPrime, allCloseAtoms, debug=debug )


    if(debug==1):
        stp.startDebugFile( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debugFileName(debug=0)
        fd = open(fDebugName, "a")
        
    #Now we will optimize and pick the  appropriate coordinate
    ##Check if all energy values are 0!!
    if(all(x == 0 for x in S[:,-1].flatten())):
      aatCoordPrime = aatPrimePositionMat[0] #first value is at 0 degrees, next at 10, then at 20, etc degrees
      stp.append2log("Taking h-bonding possibility at angle 0, as all energy values=0\n", debug=0)

      if(debug==1):
        fd.write("Taking h-bonding possibility at angle 0, as all energy values=0\n")
        fd.flush()
        stp.endDebugFile(__name__,sys._getframe().f_code.co_name, fd) 

      return aatCoordPrime, S, aatPrimePositionMat
      
    ##Find the angle where the energy is minimum
    Ind_min = np.where(S==np.min(S[:,-1]))
    Ind_min_val = Ind_min[0][0]
    
    index =  S[:,-1].argsort(axis=0)[::-1] ##Descending order according to energy value

    ##arranging S from max to min of energy values
    S = S[index]
   
    #all the attached atom prime coordinates in an array and in order from max to min
    aatPrimePositionMat = np.array(aatPrimePositionMat)[index]
    #Pick up the coordinate at the end as it is arranged in a descending order
    aatCoordPrime = aatPrimePositionMat[-1]
    minV = np.amin(S[:,-1]) #minimum energy val
    if(debug==1):
        fd.write(f"Min Energy corresponding to :{S[-1]} has coords {aatCoordPrime}\n")
        fd.write(f"Angle | Energy Sum:\n{S} \n")
        fd.write(f"Min energy found:{minV} at angle: {S[np.where(S[:,-1]==minV)[0][0]][0]}\n") 
        fd.flush()
        stp.endDebugFile(__name__,sys._getframe().f_code.co_name, fd) 

    return aatCoordPrime, S, aatPrimePositionMat


def adjustAAt0Prime(aboveSp3CoordPrime, sp3CoordPrime, aat0CoordPrime, debug=0):
    
    ''' 
     objective: Modify the initial attached atom coordinate which may be 
     from 90-120 degrees to exactly 109 degrees as per SP3 requirement

    Input: -aboveSp3CoordPrime: atom(non hydrogen/ non lone pair) connected to sp3 in prime frame
           -sp3CoordPrime: sp3 atom coord in prime frame
           -aat0CoordPrime: initial attached atom coordinates in prime frame 

    Output: aat0CoordPrimeMod: modified coordinates of the attached active atom0(Hydrogen) 

    '''

    V0 = Vector(aboveSp3CoordPrime)
    V1 = Vector(sp3CoordPrime )
    V2 = Vector(aat0CoordPrime)
    adjustTheta = mc.sp3Angle-(calc_angle(V0, V1, V2)*180/np.pi)
    aat0CoordPrimeMod = np.einsum('ij,j->i',mm.rotMatrixAboutY(adjustTheta, debug=debug), aat0CoordPrime)
    
    return aat0CoordPrimeMod


def get_closeAtomCoords(closeAtoms, debug=0):

    '''
    objective: Get close atom coords 
    input: -closeAtoms: list of close atoms in format: [atomClose, atomClose.get_parent().id[1],atomClose.get_parent().resname, aClose.get_behavior().abbrev, dist], example: [<Atom ND1>, 25, 'HIE', 'ac', 2.663196],
    output: -closeAtomCoords: list of the coordinates of each close atom

    '''
    numCloseAtoms =  np.shape(closeAtoms)[0] 
    closeAtomCoords =[]
    for i in range(1,numCloseAtoms):
        closeAtomCoords.append(closeAtoms[i][0].coord)
    return closeAtomCoords


def set_PrimeCoords(sp3, aboveSp3, allCloseAtoms, sp3CoordPrime, aboveSp3CoordPrime, closeAtomCoordsPrime, debug=0):
    ''' 
        objective: To set sp3, connected to sp3 and close atoms coordinate in the prime frame of reference
        Input:-sp3: sp3 atom
              -aboveSp3: atom(non hydrogen/lone pair) connected to sp3 atom
              -allCloseAtoms: all relevant close atoms
              -sp3CoordPrime: sp3 atom coordinate in prime frame of reference
              -aboveSp3CoordPrime: atom(non hydrogen/lone pair) connected to sp3 atom in prime frame of reference
              -closeAtomCoordsPrime: all close atom coordinates in prime frame of reference 

       Output: setting the given atom's coordinates in the prime frame of reference

    '''
        
    sp3Coord = sp3.coord
    aboveSp3Coord = aboveSp3.coord
    
    sp3.set_coord(sp3CoordPrime)
    aboveSp3.set_coord(aboveSp3CoordPrime)

    if(allCloseAtoms !=[]):
        if(np.shape(closeAtomCoordsPrime)[0]== 1 ):
            allCloseAtoms[1][0].set_coord(np.array([closeAtomCoordsPrime[0][0], closeAtomCoordsPrime[0][1], closeAtomCoordsPrime[0][2]]))
        else:
            for i in range(1, np.shape(allCloseAtoms)[0]):
                allCloseAtoms[i][0].set_coord(closeAtomCoordsPrime[i-1])


def set_OriginalCoords(sp3, aboveSp3, allCloseAtoms, sp3Coord, aboveSp3Coord, allCloseAtomCoords, debug=0):
       
     ''' 
        objective: To set sp3, connected to sp3 and close atoms coordinate in the original frame of reference
        Input:-sp3: sp3 atom
              -aboveSp3: atom(non hydrogen/lone pair) connected to sp3 atom
              -allCloseAtoms: all relevant close atoms
              -sp3CoordPrime: sp3 atom coordinate in original frame of reference
              -aboveSp3CoordPrime: atom(non hydrogen/lone pair) connected to sp3 atom in original frame of reference
              -closeAtomCoordsPrime: all close atom coordinates in original frame of reference 

       Output: setting the given atom's coordinates in the original frame of reference

'''

     sp3.set_coord(sp3Coord)
     aboveSp3.set_coord(aboveSp3Coord)

     if(allCloseAtoms !=[]):
        moveCloseAtoms = np.array(allCloseAtomCoords)
        if(len(moveCloseAtoms.shape)== 1 ):
            allCloseAtoms[1][0].set_coords(moveCloseAtoms)
        else:
            for i in range(1, np.shape(allCloseAtoms)[0]):
                allCloseAtoms[i][0].set_coord(moveCloseAtoms[i-1])




def print_closeAtomCoords(closeAtoms, codePos="default", debug=0):

    ''' 
        objective: To debug:print close atom coords at a certain code positon
        input:-closeAtoms: list of close atoms,
              -codePos: is a string to define where one is in the code
        output: prints to screen the close atom coords and where we are in the code

    '''
    numCloseAtoms =  np.shape(closeAtoms)[0] 
    closeAtomCoords =[]
    if(debug==1):
        stp.startDebugFile( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debugFileName(debug=0)
        fd = open(fDebugName, "a")
        fd.write("###############################################################################\n\n")
        fd.flush()

        for i in range(1,numCloseAtoms):
            fd.write(f"I am at: {codePos} closeAtoms:{closeAtoms[i]} coord: {closeAtoms[i][0].coord}\n")
            fd.flush()
        
        stp.endDebugFile(__name__,sys._getframe().f_code.co_name, fd) 
 

def computeAAtCoordsSp3_energy(sp3, aboveSp3, allCloseAtoms, debug=0):

     '''
        Objective: To compute optimized attached atom coordinates(Hydrogen and lone pairs) for the SP3 atom

        Input: -sp3: SP3 atom
               -aboveSP3: atom (non hydrogen/lone pair) connected to sp3 atom 
               -allCloseAtoms: the relevant close atoms around the sp3 atom

        Output: -allAAtInOriginal:all the potential attached atom coordinates in original frame of reference. The last
                                  item in the list is the optimized coordinate for the three positions(h0/h1/h2 or h0/LP1/                                  LP2). This is also easier to read/access 
                -allAAt: all the potential attached atom coordinates in original frame of reference. The last three
                                  coordinates in the list is the optimized coordinate picked. This is differently arranged                                  than allAAtInOriginal, incase we want to pick one coordinate at a time

                -coordAngEnergy: optimized coordinate and its associated angle (as its rotated to find optimum value)
                                 and its associated energy (SrevStored). It has shape: np.shape(coordAngEnergy)= (1, 2).
                                 -coordAngEnergy[0][0]: optimized coordinate value, np.shape(coordAngEnergy[0][0]) =(3,3)
                                 -coordAngEnergy[0][1]: associated angle & energy, np.shape(coordAngEnergy[0][1])= (2,1)
     '''    
    
     struct= sp3.parent.parent.parent.parent
     
     fLogName = stp.get_logFileName(debug=0)
     fLog = open(fLogName, "a")

     if(debug==1):
        stp.startDebugFile( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debugFileName(debug=0)
        fd = open(fDebugName, "a")
        


     bondLen = 1
     sp3Coord = sp3.coord
     aboveSp3Coord = aboveSp3.coord

     if(debug==1):
        fd.write(f"sp3Coord ORIGINAL: {sp3.coord} and above sp3Coord ORIGINAL: {aboveSp3.coord} \n")
        fd.flush()
        fd.close()
        print_closeAtomCoords(allCloseAtoms, codePos = "before computing initial attached atom coord:", debug=debug)
     aat0Coord = computeInitialSp3BondedAtomCoord(sp3Coord, allCloseAtoms[1][0].coord, bondLen, debug=debug)

#    ##Create the new frame of reference
     xHatPrime, yHatPrime, zHatPrime = createPrimeFrameOfReference(sp3Coord, aboveSp3Coord, aat0Coord, debug=debug)
#    ##Create the transformation matrix
     transformMatrix = setupTransformMatrix( xHatPrime, yHatPrime, zHatPrime, debug=debug)
     if(debug==1):
        fd = open(fDebugName, "a")
        fd.write(f"xP:{xHatPrime}, yP:{yHatPrime}, zP:{zHatPrime}\n")
        fd.write(f"CHECK: \nU*U^T = {np.matmul(transformMatrix, np.transpose(transformMatrix))} = I?\n")
        fd.flush()

     moveCoords = np.array((sp3Coord, aboveSp3Coord, aat0Coord)) 
     primeCoords = moveCoordsToPrimeFrameOfReference(sp3Coord, transformMatrix, moveCoords, debug=debug) 
     sp3CoordPrime = primeCoords[0]
     aboveSp3CoordPrime = primeCoords[1]
     aat0CoordPrime = primeCoords[2]

#################### Move to new frame of reference###########################

     if(allCloseAtoms):
        closeAtomCoords = get_closeAtomCoords(allCloseAtoms, debug=0)
        closeAtomCoordsPrime = moveCoordsToPrimeFrameOfReference(sp3Coord, transformMatrix, closeAtomCoords, debug=debug)

     set_PrimeCoords(sp3, aboveSp3, allCloseAtoms, sp3CoordPrime, aboveSp3CoordPrime, closeAtomCoordsPrime, debug=debug)
##########################In the new frame ####################################################################
     aat0CoordPrimeMod = adjustAAt0Prime(aboveSp3.coord, sp3.coord, aat0CoordPrime, debug=debug)
     if(debug == True):
        fd.write(f"SP3 Prime:{primeCoords[0]} \n \
                ABOVE SP3 Prime:{primeCoords[1]} ")

        fd.write(f"Is the y component of h0_prime: {aat0CoordPrime} = [X' comp, 0, Z' comp]?\n") 
        fd.write(f"Is the y component of h0_prime_modified for adjusted angle to get 109.5: {aat0CoordPrimeMod} = [X' comp, 0, Z' comp]? \n")
        fd.flush()
        V0 = Vector(aboveSp3CoordPrime)
        V1 = Vector(sp3CoordPrime )
        V2 = Vector(aat0CoordPrime)
        V3 = Vector(aat0CoordPrimeMod)
        fd.write(f"Angle before angular adjustment to reach 109.5:{calc_angle(V0,V1,V2)*180/np.pi} and angle now: {calc_angle(V0,V1,V3)*180/np.pi}\n")
        fd.flush()

     aatPrimeDistMin, S, aatPrimeMat = computeAAtInNewFrameOfReference_energy(sp3, aboveSp3, allCloseAtoms, aat0CoordPrimeMod, debug=debug)

#########################End the new frame of reference####################################################################
     set_OriginalCoords(sp3, aboveSp3, allCloseAtoms, sp3Coord, aboveSp3Coord, closeAtomCoords, debug=debug)
     if(debug==True):
        fd.write(f"back to original coords,sp3Coord: {sp3.coord}, above sp3Coord: {aboveSp3.coord}\n")
        fd.flush()
        fd.close()
        print_closeAtomCoords(allCloseAtoms, codePos = "back to original frame of reference", debug=debug)
        checkBondAngForAllAAt( sp3, aboveSp3, sp3CoordPrime, aboveSp3CoordPrime, aatPrimeDistMin[0], aatPrimeDistMin[1], aatPrimeDistMin[2], debug=debug) 
        fd = open(fDebugName, "a")
#################### Get back to original frame of reference ONLY the MIN SUM DIST###########################
    
     numRows = np.shape(aatPrimeMat)[0]*np.shape(aatPrimeMat)[1]
     
     allAAt = moveCoordsToOriginalFrameOfReference(sp3Coord, transformMatrix, np.reshape(aatPrimeMat,(numRows,3)), debug =0) 
     numAAt = int(np.shape(allAAt)[0]/3)
     allAAtInOriginal = np.reshape(allAAt,(numAAt,3,3))
   
     coordAngEnergy = []
     Srev = S[::-1]
    
######Check if the optimized coordinate picked clashes with any surrounding atoms, if yes: pick next best option#######################
     for i,AAtInOriginal in reversed(list(enumerate(allAAtInOriginal))):
         #since LYS has 3 attached hydrogens that need to be checked, else only one hydrogen is in SER/THR.
         #so all the hydrogen needs to be checked for  the clash. Lone pairs are only pseudo atoms-so we do not check that

         hMax=3 if sp3.parent.resname == 'LYS' else 1
        

         for j in range(hMax):
             coord2check  = AAtInOriginal[j]
             fLog.write(f"Need to look for coord so H does not clash. max hyd={hMax}. Currently at iteration:i:{i} \n")
             fLog.flush()

             if(debug==1):
                fd.write(f"Need to look for coord so H does not clash. max hyd={hMax}. Currently at iteration:i:{i} \n")
                fd.flush()


             residue = sp3.parent
             structure = residue.parent.parent.parent

             resSearchList  = stp.get_allResidues(structure, donotIncludeRes = residue, debug=debug)

             searchListAtoms = []
             for res in resSearchList:
                    searchListAtoms.append(stp.createListOfAtomsOfResidueWithoutLP(res, debug=debug))

             ##Flatten it
             searchListAtoms =  [atom for atomlist in searchListAtoms for atom in atomlist]
        #    ###use the nbd search and create an object ns.
             ns = Bio.PDB.NeighborSearch(searchListAtoms)
             potClashAtom = ns.search(coord2check,1.5)

             if(potClashAtom):
                fLog.write(f"potential clash atom = {potClashAtom} \n")
                fLog.flush()
                if(debug==1):
                    fd.write(f"potential clash atom = {potClashAtom} \n")
                    fd.flush()

                for pcAtom in potClashAtom:
                    fLog.write(f"orignal atom: pot Hyd, its parent: {sp3.parent}, pcAtom:{pcAtom} and parent: {pcAtom.parent} and distance is: {np.linalg.norm(coord2check-np.float32(pcAtom.coord))} \n")
                    fLog.flush()
                    if(debug==1):
                        fd.write(f"orignal atom: pot Hyd, its parent: {sp3.parent}, pcAtom:{pcAtom} and parent: {pcAtom.parent} and distance is: {np.linalg.norm(coord2check-np.float32(pcAtom.coord))} \n")
                        fd.flush()
                continue
             else: 
                if(j != hMax-1):
                    continue
                else:
                    SrevStored=Srev[j].reshape(2,1) #first in the list is angle, the other is energy
                    coordAngEnergy.append( [AAtInOriginal, SrevStored])
                    allAAt = np.append(allAAt, AAtInOriginal, axis=0)
                    allAAtInOriginal = np.append(allAAtInOriginal, [AAtInOriginal], axis =0)
                    if(debug==1):
                        if(sp3.parent.resname=='LYS'): 
                            aroundSp3 ='Hydrogen/Hydrogen/Hydrogen'
                        else:
                            aroundSp3 ='Hydrogen/lone pair/lone pair'
                         
                        if(debug==1):
                            fd.write(f"\n Coordinates picked for {aroundSp3}: {coordAngEnergy[0][0]},\n at angle:{ coordAngEnergy[0][1][0]} with energy: {coordAngEnergy[0][1][1]} \n")
                            fd.flush()
                            stp.endDebugFile(__name__,sys._getframe().f_code.co_name, fd) 

                    fLog.close()
                    return allAAtInOriginal, allAAt, coordAngEnergy



def placeHydrogens_SER_THR(res, lastSerial, debug=0):

    ''' 
        objective: placing the hydrogens and Lone pairs for Serine 

        Input: -res: residue(SER/THR) that requires hydrogen and lone pair placement 
               -lastSerial: the serial number of the last atom in the structure

        Output: -updatedLastSerial: updated serial number of the added atom
                -aatCoords : attached atom (Hydrogen and lone pairs) coordinates
    '''
    fLogName = stp.get_logFileName(debug=0)
    fLog = open(fLogName, "a")

    fLog.write(f"\n***********************************************************************************************\n\n")
    fLog.write(f"Place hydrogen for for {res}, {res.id[1]} of chain: {res.parent} \n")
    fLog.flush()

    if(debug==1):
        stp.startDebugFile( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debugFileName(debug=0)
        fd = open(fDebugName, "a")
        fd.write(f"\n***********************************************************************************************\n\n")
        fd.write(f"Place hydrogen for for {res} of chain: {res.parent} \n")
        fd.flush()
    
    structure = res.parent.parent.parent

    aboveSp3 = res['CB']

    if(res.resname == 'SER'): 
        sp3 = res['OG']
    else: 
        sp3 = res['OG1']

    customList = stp.get_knownDonorAcceptorListWRTOneAtom(structure,sp3, aaType = 'DONOR_ACCEPTOR', debug=debug)

    if(not customList):
        fLog.write(f"NO KNOWN CLOSE ATOMS FOUND for {res}, {res.id[1]} and chain: {res.parent}\n")
        fLog.flush()
        updatedLastSerial = lastSerial
        aatCoords = []
        fLog.close()
        return updatedLastSerial, aatCoords

    ##Get a list of all close Atoms (which are donor, acceptor, and both) for the given hvy atom
    allCloseAtoms = cats.get_allCloseAtomInfoForOneAtom(sp3, customList, debug=debug)

    
    if(np.shape(allCloseAtoms)[0] == 1):
        fLog.write(f"NO CLOSE ATOMS FOUND for {res}, {res.id[1]} of chain: {res.parent}\n")
        fLog.flush()
        if(debug==1):
            fd.write(f"NO CLOSE ATOMS FOUND for {res}, {res.id[1]} of chain: {res.parent}\n")
            fd.flush()
        updatedLastSerial = lastSerial
        aatCoords = []
        fLog.close()
        return updatedLastSerial, aatCoords

    if(debug==1):
        fd.close()
    aatInOriginal, aatInOriginalAppended, coordAngEnergy = computeAAtCoordsSp3_energy(sp3, aboveSp3, allCloseAtoms, debug=debug)
    aatCoords = aatInOriginal[-1]

    names = ['HG', 'LP3', 'LP4'] if(res.resname == 'SER') else ['HG1', 'LP3', 'LP4']
    elements = ['H','LP','LP']

    #add atoms to reside:
    for i in range(len(names)):

        res.add(Bio.PDB.Atom.Atom(name=names[i], coord=aatCoords[i], bfactor=0., occupancy=1., altloc=' ', fullname=names[i], serial_number=lastSerial+i,element=elements[i]))
    fLog.write(f"Side chain hydrogen placed, for {res} with chain: {res.parent}, added {names}, with s.no now at: {lastSerial+i} !\n")
    fLog.write(f"\n***********************************************************************************************\n\n")
    fLog.flush()

    if(debug==1):
        fd = open(fDebugName, "a")
        fd.write(f"Side chain hydrogen placed, for {res} with chain:{res.parent}, added {len(names)} Hs, with s.no now at: {lastSerial+i} !\n")
        fd.write(f"\n***********************************************************************************************\n\n")
        fd.flush()

        stp.endDebugFile(__name__,sys._getframe().f_code.co_name, fd) 

    updatedLastSerial = lastSerial + i

    fLog.close()
    return updatedLastSerial, aatCoords


def placeHydrogens_LYS(res, lastSerial, debug=0):
    '''
    objective: placing the hydrogen and Lone pairs for Lysine

        Input: -res: residue(LYS) that requires hydrogen placement
               -lastSerial: the serial number of the last atom in the structure

        Output: -updatedLastSerial: updated serial number of the added atom
                -hhCoords : three hydrogen coordinates attached to NZ

    '''

    fLogName = stp.get_logFileName(debug=0)
    fLog = open(fLogName, "a")
    fLog.write(f"\n***********************************************************************************************\n\n")
    fLog.write(f"Place hydrogen for for {res} of chain: {res.parent} \n")
    fLog.flush()
  
    if(debug==1):
        stp.startDebugFile( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debugFileName(debug=0)
        fd = open(fDebugName, "a")
        fd.write(f"\n***********************************************************************************************\n\n")
        fd.write(f"Place hydrogen for for {res} of chain: {res.parent} \n")
        fd.flush()

    aboveSp3 = res['CE']
    sp3 = res['NZ']
    structure = res.parent.parent.parent
    ###Get only the known atoms for generating close atoms!!!
    customList = stp.get_knownDonorAcceptorListWRTOneAtom(structure,sp3, aaType = 'DONOR_ACCEPTOR', debug=debug)


    if(debug==1):
        stp.startDebugFile( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debugFileName(debug=0)
        fd = open(fDebugName, "a")

        fd.write(f"\n***********************************************************************************************\n\n")
        fd.write(f"Place hydrogen for for {res} of chain: {res.parent} \n")
        fd.flush()

    if(not customList):
        fLog.write(f"NO KNOWN CLOSE ATOMS FOUND for {res}, {res.id[1]} of chain: {res.parent}\n")
        fLog.flush()

        if(debug==1):
            fd.write(f"NO KNOWN CLOSE ATOMS FOUND for {res}, {res.id[1]} of chain: {res.parent}\n")
            fd.flush()

        hCoords = []
        updatedLastSerial = lastSerial
        fLog.close()
        return updatedLastSerial, hCoords

    ##Get a list of all close Atoms (which are donor and  acceptor) for the given hvy atom
    allCloseAtoms = cats.get_allCloseAtomInfoForOneAtom(sp3, customList, debug=debug)

    if(np.shape(allCloseAtoms)[0] == 1):
        fLog.write(f"NO CLOSE ATOMS FOUND for {res}, {res.id[1]} of chain: {res.parent}\n")
        fLog.flush()

        if(debug==1):
            fd.write(f"NO CLOSE ATOMS FOUND for {res}, {res.id[1]} of chain: {res.parent}\n")
            fd.flush()
        hCoords = []
        updatedLastSerial = lastSerial
        fLog.close()
        return updatedLastSerial, hCoords   

    if(debug==1):
        fd.close()
    #optimize and relevant coordinates of the hydrogens, associates angle and energy!!
    aatInOriginal, aatInOriginalAppended, coordAngEnergy = computeAAtCoordsSp3_energy(sp3, aboveSp3, allCloseAtoms,debug=debug)
    hCoords = aatInOriginal[-1]

    names = ['HZ1', 'HZ2', 'HZ3']
    
    #add the atoms to the residue
    for i in range(len(names)):
        res.add(Bio.PDB.Atom.Atom(name=names[i], coord=hCoords[i], bfactor=0., occupancy=1., altloc=' ', fullname=names[i], serial_number=lastSerial+i,element='H'))

    fLog.write(f"Side chain hydrogen placed, for {res} with chain:{res.parent}, added {len(names)} Hs, with s.no now at: {lastSerial+i} !\n")
    fLog.write(f"\n***********************************************************************************************\n\n")
    fLog.flush()
    if(debug==1):
        fd = open(fDebugName, "a")
        fd.write(f"Side chain hydrogen placed, for {res} with chain:{res.parent}, added {len(names)} Hs, with s.no now at: {lastSerial+i} !\n")
        fd.write(f"\n***********************************************************************************************\n\n")
        fd.flush()

        stp.endDebugFile(__name__,sys._getframe().f_code.co_name, fd) 

    updatedLastSerial = lastSerial+i
   
    fLog.close()
    return updatedLastSerial, hCoords


def addSP3SideChainHydrogens(structure, lastSerial):

    ''' Objective: To add all SP3 Hydrogen SideChains '''

    res_dict = {'LYS': placeHydrogens_LYS, 'SER':placeHydrogens_SER_THR, 'THR': placeHydrogens_SER_THR}

    for res in structure.get_residues():
        try:
            lastSerial, hCoord = res_dict[res.resname](res, lastSerial)
        except KeyError:
            stp.append2log(f"Not SP3/insufficient Atoms: {res.resname} with ID: {res.id[1]} and chain: {res.parent}\n", debug=0 )
            pass
            
