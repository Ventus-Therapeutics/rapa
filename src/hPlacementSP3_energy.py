import os
import sys
import Bio
#from sympy import *
import numpy as np
#from collections import namedtuple
from Bio.PDB import *

import code
import math
#import csv
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D 

import myResAtom as mra
import myConstants as mc
import setupProt as stp
import hPlacementSP2 as hsp2
import closeAtoms as cats
import myMath as mm

def checkBondAngForAllAAt(sp3, abvSP3, aat0, aat1, aat2):

    ''' Objective: checks/prints the bond angle with all attached atoms
        Input:-sp3: sp3 atom
              -abvSP3: atom connected with sp3 that is not hydrogen or LP
              -aat0, aat1, aat2: Hyd/LP atoms
              
        Output: Prints out the angle between above SP3 atom-SP3 atom-hyd/LP atom

    '''
    sp3Vec = Vector(sp3)
    abvSP3Vec = Vector(abvSP3)
    aat0Vec =Vector(aat0)
    aat1Vec =Vector(aat1)
    aat2Vec =Vector(aat2)
    stp.append2log(f"Calc angle:aboveSP3-SP3-Hcoord0 {calc_angle(abvSP3Vec, sp3Vec, aat0Vec)*180/3.14}, \n")
    stp.append2log(f"Calc angle:aboveSP3-SP3-Hcoord1/LP1 {calc_angle(abvSP3Vec, sp3Vec, aat1Vec)*180/3.14}, \n")
    stp.append2log(f"Calc angle:aboveSP3-SP3-Hcoord2/LP2 {calc_angle(abvSP3Vec, sp3Vec, aat2Vec)*180/3.14}, \n")



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
    sp3AAt0 = bondLength*mm.unitVec(surrCoord-sp3Coord) 
    #Find the position vector
    aat0PosVec = sp3AAt0 + sp3Coord
    if(debug ==True):
        print(f"The initial sp3 bonded atom coord is: {aat0PosVec}")

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


    zHatPrime = mm.unitVec(aboveSp3Coord - sp3Coord)
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

    xHatPrime = mm.unitVec(xPrime)
    yHatPrime = np.cross(zHatPrime, xHatPrime)

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
    translatedCoords = mm.translateCoord(shiftByCoord, allCoordsToMove)
    ###Now doing a matmult or einsum to get transformedCoords = np.matmul(transformMatrix, translatedCoords)
    if(len(translatedCoords.shape)== 1):
        transformedCoords = np.einsum('ij,kj->ki', transformMatrix, [translatedCoords] )
    else:
        transformedCoords = np.einsum('ij,kj->ki', transformMatrix, translatedCoords ) 

    return transformedCoords



def moveCoordsToOriginalFrameOfReference(shiftCoord, transformMatrix, moveCoordsBack, debug =0):

    ''' Objective: Move the desired coordinates to the original frame of reference from new frame of reference 
        Input: -shiftCoord/sp3Coord: Coordinates to translate or shift by (coordinate of sp3 atoms)
               -transformMatrix: Transform or Rotation matrix to get back to original frame of ref(xprime,yprime,zprime)
               -moveCoordsBack: Coords to move
        Output:-hBack: original/Transformed to original coordinate of the moveCoordsBack 

    '''
    hBack = np.einsum("ij,kj->ki", np.linalg.inv(transformMatrix), moveCoordsBack) + shiftCoord    
    return hBack

def getAllAttachedAtoms(aat0CoordPrime, debug =0): 

    ''' 
        Objective: Get the other two attached atom coords given one attached atom coord in the new frame by rotating   
        Input: -aat0CoordPrime: coordinate of the first attached atom of the sp3 atom
        Output: -aatPrime: array of the three attached atom to the sp3 atom that is 109.5 degrees apart
    '''

    rotInPlane = np.array((mm.rotMatrix(120),mm.rotMatrix(240)))

    aat1CoordPrime, aat2CoordPrime = np.einsum('ijk,k->ij',rotInPlane, aat0CoordPrime)
    aatPrime = np.array((aat0CoordPrime, aat1CoordPrime, aat2CoordPrime))
  
    if(debug == True):
        checkBondAngForAllAAt(SP3, abvSP3, aat0CoordPrime, aat1CoordPrime, aat2CoordPrime)

    return aatPrime 

def get_threeAatEnergy(closeAt, sp3, aboveSp3, aatPrime, attractive = [1,1,1,0,0]):
    
    '''
    Get the three energies
    '''

    behav = myCloseAt.get_behavior().abbrev
    if(debug == True):
        print(f"closeAT: {closeAt.coord}, sp3:{sp3.coord} aboveSp3: {aboveSp3.coord}, aatPrime:{aatPrime}")
        print(f"close atom is ACCEPTOR i:{i} with close atom as: {closeAt} and sp3:{sp3} and its parent: {sp3.parent}")

    ##Assume the first to be the Hydrogen atom (the others can be either two hydrogens-LYS or two lone pairs-SER/THR)
    aat0Energy, aat0enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[0], sp3, attractive=attractive[0], atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)

    if(debug == True):
        print(f"I am {sp3} with parent {sp3.parent.resname} CP:{closeAt}, behv:{behav}, action: ATTRACT  aat0En: {aat0Energy}, aat0SumR = {aat0enSum}")

    if(sp3.parent.resname == 'LYS'):
        #For the case of two hydrogens:
        aat1Energy, aat1enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[1], sp3,  attractive=attractive[1], atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)
        aat2Energy, aat2enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[2], sp3,  attractive=attractive[2], atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)
    else:
        aat1Energy, aat1enSum = cats.computeEnergyAsAcceptor(sp3, Vector(aatPrime[1]), closeAt, attractive = attractive[3], atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)
        aat2Energy, aat2enSum = cats.computeEnergyAsAcceptor(sp3, Vector(aatPrime[2]), closeAt, attractive = attractive[4], atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)
        

    if(debug == True):
        print(f"I am {sp3} with parent {sp3.parent.resname} CP:{closeAt}, behv:{behav}, is attractive: {attractive}")
        print(f"CP:ACC aat0En: {aat0Energy}, aat0Sum = {aat0enSum}")
        print(f"CP:ACC aat1En: {aat1Energy}, aat1Sum = {aat1enSum}")
        print(f"CP:ACC aat2En: {aat2Energy}, aat2Sum = {aat2enSum}")

    return aat0Energy, aat0enSum, aat1Energy, aat1enSum, aat2Energy, aat2enSum

    
####################################################################################################
def energyValOfACATSforHLP(sp3, aboveSp3, aatPrime, allCloseAtoms, ang, debug=0):

    '''
        objective: To find energy value of all close atoms for Hyd-0/Hyd-1/Hyd-2  is Sp3.parent ==LYS
                    or  Hyd-0/LP-1/LP-2 if sp3.parent == SER/THR

        input:-sp3: Sp3  atom
              -aboveSp3: atom connected to sp3 which is not Hydrogen/lone pair
              -aatPrime: active atom prime coordinates (Hyd/lone pair)
              -allCloseAtoms: All close atoms
              -ang: angle at which energy values are being computed

        output: -aat0Energy, aat1Energy, aat2Energy: energies wrt  Hyd-0/Hyd-1/Hyd-2 or Hyd-0/LP-1/LP-
                -enSum: sum of the energies 

    '''

    if(debug == True): 
        print(f"In funct: energyValOfACATSforHLP, close Atoms: {allCloseAtoms}") 
    #energy summation
    enSum = 0
    ##Additional information for debug purpose:ang, sp3.parent.resname, sp3.parent.id[1], closeAt.parent.resname, closeAt.parent.id[1], aat0Energy, aat1Energy, aat2Energy, enSum

    energySumsInfo =[]

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
            if(debug == True): print(f"close atom is ACCEPTOR i:{i} with close atom as: {closeAt} and sp3:{sp3} and its parent: {sp3.parent}")

            ##Assume the first to be the Hydrogen atom (the others can be either two hydrogens-LYS or two lone pairs-SER/THR)
            aat0Energy, aat0enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[0], sp3, attractive=1, atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)

            if(debug == True):
                print(f"I am {sp3} with parent {sp3.parent.resname} CP:{closeAt}, behv:{behav}, action: ATTRACT  aat0En: {aat0Energy}, aat0SumR = {aat0enSum}")

            if(sp3.parent.resname == 'LYS'):
                #For the case of two hydrogens:
                aat1Energy, aat1enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[1], sp3,  attractive=1, atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)
                aat2Energy, aat2enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[2], sp3, attractive=1, atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)

            else:
                aat1Energy, aat1enSum = cats.computeEnergyAsAcceptor(sp3, Vector(aatPrime[1]), closeAt, attractive = 0, atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)
                aat2Energy, aat2enSum = cats.computeEnergyAsAcceptor(sp3, Vector(aatPrime[2]), closeAt, attractive = 0, atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)
                
            enSum = enSum + aat0enSum + aat1enSum + aat2enSum

            if(debug == True):
                print(f"I am {sp3} with parent {sp3.parent.resname} CP:{closeAt}, behv:{behav}, is attractive: {attractive}")
                print(f"CP:ACC aat0En: {aat0Energy}, aat0Sum = {aat0enSum}")
                print(f"CP:ACC aat1En: {aat1Energy}, aat1Sum = {aat1enSum}")
                print(f"CP:ACC aat2En: {aat2Energy}, aat2Sum = {aat2enSum}")
                print(f"CP:ACC energy Sum = {enSum}")
            ########################################################

        if(myCloseAt.get_behavior().abbrev == 'do'):
            if(debug == True):
                print(f"closeAT: {closeAt.coord}, sp3:{sp3.coord} aboveSp3: {aboveSp3.coord}, aatPrime:{aatPrime}")
                print(f"close atom is DON i:{i} with close atom as: {closeAt} and sp3:{sp3} and its parent: {sp3.parent}")

            ##Assume the first to be the Hydrogen atom (the others can be either two hydrogens-LYS or two lone pairs-SER/THR)
            aat0Energy, aat0enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[0], sp3, attractive=0, atype= 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)

            if(debug == True):
                print(f"I am {sp3} with parent {sp3.parent.resname} CP:{closeAt}, behv:{behav}, action: ATTRACT  aat0En: {aat0Energy}, aat0SumR = {aat0enSum}")

            if(sp3.parent.resname == 'LYS'):
                #For the case of two hydrogens:
                aat1Energy, aat1enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[1], sp3, attractive =0, atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)
                aat2Energy, aat2enSum = cats.computeEnergyAsDonor(closeAt, aatPrime[2], sp3, attractive =0, atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)

            else:
                aat1Energy, aat1enSum = cats.computeEnergyAsAcceptor(sp3, Vector(aatPrime[1]), closeAt, attractive = 1, atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)
                aat2Energy, aat2enSum = cats.computeEnergyAsAcceptor(sp3, Vector(aatPrime[2]), closeAt, attractive = 1, atype = 'SP3',  chV_levelVal ='level_00_chV_00_sNum_00', debug = 0)

            enSum = enSum + aat0enSum + aat1enSum + aat2enSum

            if(debug == True):
                print(f"I am {sp3} with parent {sp3.parent.resname} CP:{closeAt}, behv:{behav}, is attractive: {attractive}")
                print(f"CP:ACC aat0En: {aat0Energy}, aat0Sum = {aat0enSum}")
                print(f"CP:ACC aat1En: {aat1Energy}, aat1Sum = {aat1enSum}")
                print(f"CP:ACC aat2En: {aat2Energy}, aat2Sum = {aat2enSum}")
                print(f"CP:ACC energy Sum = {enSum}")

        energySumsInfo.append([ang, sp3.parent.resname, sp3.parent.id[1], closeAt.parent.resname, closeAt.parent.id[1], aat0Energy, aat1Energy, aat2Energy, enSum])
        if(debug == True): 
            print(f"EnergySums ={energySums}")

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
    if(debug == True):
        print(f"\n \n Initial h0,h1, h2: { aatPrime[0]}, { aatPrime[1]},{ aatPrime[2]} \n \n")
    sumEnergyInfo = [] 
    aatPrimeMat = []
    aatSurrAllAngles = []
    #end angle for LYS: 120, and end angle for SER/THR: 360
    endAng = mc.optRotAngle_LYS if(sp3.parent.resname == 'LYS') else mc.optRotAngle
    
    for ind,ang in enumerate(range(0,endAng,10)):
        aatPrimeRot = np.einsum("ij,kj->ki", mm.rotMatrix(ang), aatPrime)

        if(debug == True):
            print(f" hp:{aatPrimeRot[0]},{aatPrimeRot[1]},{aatPrimeRot[2]} rotated:{ang}")
            checkBondAngForAllAAt(sp3.coord, aboveSp3.coord, aatPrimeRot[0], aatPrimeRot[1], aatPrimeRot[2] )
        aatPrimeMat.append(aatPrimeRot)   

        ##aat1 and aat2 will be acceptors if SER/THR or aa1 and aat2 will be donors if LYS #ACATS = All Close Atoms
        aat0Energy, aat1Energy, aat2Energy, enSum = energyValOfACATSforHLP(sp3, aboveSp3, aatPrimeMat[ind], allCloseAtoms, ang, debug =0)               
        sumEnergyInfo.append([ang, enSum])
        if(debug == True):
            print(f"sum Energy Info: {sumEnergyInfo}")

    #outputting it as an array
    S = np.array(sumEnergyInfo)
    if(debug == True):
        print(f"sum Energy Info: {S}")

    return S, aatPrimeMat


##############################################################################################################
def computeAAtInNewFrameOfReference_energy(sp3, aboveSp3, allCloseAtoms, aat0CoordPrime, debug = 0):
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
    aatPrime = getAllAttachedAtoms(aat0CoordPrime, debug=0) 
    #############################################################################
    #Find the coordinates for hydrogen and lone pair placement by rotating in plane and computing S i.e angle and energy computed at that angle, along with aatPrimePositionMat or attached atom coordinates
    S, aatPrimePositionMat  = optimizeByRotationInPlane_energy(sp3, aboveSp3, aatPrime, allCloseAtoms, debug =0 )

    #Now we will optimize and pick the  appropriate coordinate
    ##Check if all energy values are 0!!
    if(all(x == 0 for x in S[:,-1].flatten())):
      aatCoordPrime = aatPrimePositionMat[0]
      stp.append2log("Taking h-bonding possibility at angle 0, as all energy values=0\n")
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
    
    if(debug == True):
        print(f"Min Energy corresponding to :{S[-1]} has coords {aatCoordPrime}")
        print(f"Distance Sum:{S}")
        print(f"Min dist found:{np.amin(S[:,-1])}")

    return aatCoordPrime, S, aatPrimePositionMat


def adjustAAt0Prime(aboveSp3CoordPrime, sp3CoordPrime, aat0CoordPrime, debug =0):
    
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
    adjustTheta = 109-(calc_angle(V0, V1, V2)*180/np.pi)
    aat0CoordPrimeMod = np.einsum('ij,j->i',mm.rotMatrixAboutY(adjustTheta), aat0CoordPrime)
    
    return aat0CoordPrimeMod


def get_closeAtomCoords(closeAtoms, debug =0):

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


def set_PrimeCoords(sp3, aboveSp3, allCloseAtoms, sp3CoordPrime, aboveSp3CoordPrime, closeAtomCoordsPrime, debug = 0):
    ##IF NO CLOSE ATOM-NO POINT DOING ALL THIS!! 
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


###############################################################################################
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



def set_AtomCoords(sp3, aboveSp3, allCloseAtoms, sp3CoordToSet, aboveSp3CoordToSet, closeAtomCoordsToSet, debug = 0):
    ##This would be a general function as opposed to set_originalCoords/set_PrimeCoords
    ''' 
        objective: To set sp3, connected to sp3 and close atoms coordinate in the another frame of reference
        Input:-sp3: sp3 atom
              -aboveSp3: atom(non hydrogen/lone pair) connected to sp3 atom
              -allCloseAtoms: all relevant close atoms
              -sp3CoordToSet: sp3 atom coordinate in prime/original frame of reference
              -aboveSp3CoordToSet: atom(non hydrogen/lone pair) connected to sp3 atom in prime/orig frame of reference
              -closeAtomCoordsToSet: all close atom coordinates in prime/orig frame of reference 

       Output: setting the given atom's coordinates in the prime/orig frame of reference

    '''
        
    
    sp3.set_coord(sp3CoordToSet)
    aboveSp3.set_coord(aboveSp3CoordToSet)

    if(allCloseAtoms !=[]):
        if(np.shape(closeAtomCoordsToSet)[0]== 1 ):
            allCloseAtoms[1][0].set_coord(np.array([closeAtomCoordsToSet[0][0], closeAtomCoordsToSet[0][1], closeAtomCoordsToSet[0][2]]))
        else:
            for i in range(1, np.shape(allCloseAtoms)[0]):
                allCloseAtoms[i][0].set_coord(closeAtomCoordsToSet[i-1])


def print_closeAtomCoords(closeAtoms, codePos="default"):

    ''' 
        objective: To debug:print close atom coords at a certain code positon
        input:-closeAtoms: list of close atoms,
              -codePos: is a string to define where one is in the code
        output: prints to screen the close atom coords and where we are in the code

    '''
    numCloseAtoms =  np.shape(closeAtoms)[0] 
    closeAtomCoords =[]
    stp.append2log("###############################################################################\n\n")
    for i in range(1,numCloseAtoms):
        stp.append2log(f"I am at: {codePos} closeAtoms:{closeAtoms[i]} coord: {closeAtoms[i][0].coord}\n")

def computeAAtCoordsSp3_energy(sp3, aboveSp3, allCloseAtoms, debug = 0):

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

     fLogName = stp.get_logFileName(sp3.parent.parent.parent.parent.id)
     fLog = open(fLogName, "a")


     bondLen = 1
     sp3Coord = sp3.coord
     aboveSp3Coord = aboveSp3.coord

     if(debug ==True):
        print(f"sp3Coord ORIGINAL: {sp3.coord}")
        print(f"above sp3Coord ORIGINAL: {aboveSp3.coord}")
        print_closeAtomCoords(allCloseAtoms, codePos = "before computing initial attached atom coord:")
     aat0Coord = computeInitialSp3BondedAtomCoord(sp3Coord, allCloseAtoms[1][0].coord, bondLen, debug=0)

#    ##Create the new frame of reference
     xHatPrime, yHatPrime, zHatPrime = createPrimeFrameOfReference(sp3Coord, aboveSp3Coord, aat0Coord, debug=0)
#    ##Create the transformation matrix
     transformMatrix = setupTransformMatrix( xHatPrime, yHatPrime, zHatPrime, debug =0)
     if(debug == True):
        print(f"xP:{xHatPrime}, yP:{yHatPrime}, zP:{zHatPrime}")
        print(f"New Acceptor0 start with acceptor0 ={allCloseAtoms[1][0].coord}")
        print(f"CHECK: \nU*U^T = {np.matmul(transformMatrix, np.transpose(transformMatrix))} = I?")

     moveCoords = np.array((sp3Coord, aboveSp3Coord, aat0Coord)) 
     primeCoords = moveCoordsToPrimeFrameOfReference(sp3Coord, transformMatrix, moveCoords, debug=0) 
     sp3CoordPrime = primeCoords[0]
     aboveSp3CoordPrime = primeCoords[1]
     aat0CoordPrime = primeCoords[2]

#################### MOVE TO NEW FRAME OF REF###########################

     if(allCloseAtoms):
        closeAtomCoords = get_closeAtomCoords(allCloseAtoms, debug=0)
        closeAtomCoordsPrime = moveCoordsToPrimeFrameOfReference(sp3Coord, transformMatrix, closeAtomCoords, debug=0)

     set_PrimeCoords(sp3, aboveSp3, allCloseAtoms, sp3CoordPrime, aboveSp3CoordPrime, closeAtomCoordsPrime, debug=0)
##########################IN the new frame!!####################################################################
     aat0CoordPrimeMod = adjustAAt0Prime(aboveSp3.coord, sp3.coord, aat0CoordPrime, debug =0)
     if(debug == True):
        print(f"SP3 Prime:{primeCoords[0]} ")
        print(f"ABOVE SP3 Prime:{primeCoords[1]} ")
        print(f"Is the y component of h0_prime: {aat0CoordPrime} = [X' comp, 0, Z' comp]?") 
        print(f"Is the y component of h0_prime: {aat0CoordPrimeMod} = [X' comp, 0, Z' comp]?")
        V0 = Vector(aboveSp3CoordPrime)
        V1 = Vector(sp3CoordPrime )
        V2 = Vector(aat0CoordPrime)
        V3 = Vector(aat0CoordPrimeMod)
        print(f"Angle before:{calc_angle(V0,V1,V2)*180/np.pi} and angle now: {calc_angle(V0,V1,V3)*180/np.pi}")

     aatPrimeDistMin, S, aatPrimeMat = computeAAtInNewFrameOfReference_energy(sp3, aboveSp3, allCloseAtoms, aat0CoordPrimeMod, debug = 0)
     set_OriginalCoords(sp3, aboveSp3, allCloseAtoms, sp3Coord, aboveSp3Coord, closeAtomCoords, debug=0)
     if(debug ==True):
        print(f"sp3Coord: {sp3.coord}")
        print(f"above sp3Coord: {aboveSp3.coord}")
        print_closeAtomCoords(allCloseAtoms, codePos = "back to ORIG")
        print(f"aatPrimeDist: {aatPrimeDistMin}")
        checkBondAngForAllAAt( sp3CoordPrime, aboveSp3CoordPrime, aatPrimeDistMin[0], aatPrimeDistMin[1], aatPrimeDistMin[2])

#########################END the new frame!!####################################################################
    ##################################################################################### 
#################### Get back to original frame of reference ONLY the MIN SUM DIST###########################
    
    ####Appending the first time-is this needed?
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

             residue = sp3.parent
             structure = residue.parent.parent.parent

             resSearchList  = stp.get_allResidues(structure, donotIncludeRes = residue)

             searchListAtoms = []
             for res in resSearchList:
                    searchListAtoms.append(stp.createListOfAtomsOfResidueWithoutLP(res))

             ##Flatten it
             searchListAtoms =  [atom for atomlist in searchListAtoms for atom in atomlist]
        #    ###use the nbd search and create an object ns.
             ns = Bio.PDB.NeighborSearch(searchListAtoms)
             potClashAtom = ns.search(coord2check,1.5)

             if(potClashAtom):
                fLog.write(f"potential clash atom = {potClashAtom} \n")
                fLog.flush()

                for pcAtom in potClashAtom:
                    fLog.write(f"orignal atom: pot Hyd, its parent: {sp3.parent}, pcAtom:{pcAtom} and parent: {pcAtom.parent} and distance is: {np.linalg.norm(coord2check-np.float32(pcAtom.coord))} \n")
                    fLog.flush()
                continue
             else: 
                if(j != hMax-1):
                    continue
                else:
                    SrevStored=Srev[j].reshape(2,1) #first in the list is angle, the other is energy
                    coordAngEnergy.append( [AAtInOriginal, SrevStored])
                    allAAt = np.append(allAAt, AAtInOriginal, axis =0)
                    allAAtInOriginal = np.append(allAAtInOriginal, [AAtInOriginal], axis =0)
                    if(debug ==True):
                        print(f"CoordAngEnergy: {coordAngEnergy}")
                    fLog.close()
                    return allAAtInOriginal, allAAt, coordAngEnergy



def placeHydrogens_SER_THR(res, lastSerial, method = 'energy', debug =0):

    ''' 
        objective: placing the hydrogens and Lone pairs for Serine 

        Input: -res: residue(SER/THR) that requires hydrogen and lone pair placement 
               -lastSerial: the serial number of the last atom in the structure

        Output: -updatedLastSerial: updated serial number of the added atom
                -aatCoords : attached atom (Hydrogen and lone pairs) coordinates
    '''

    fLogName = stp.get_logFileName(res.parent.parent.parent.id)
    fLog = open(fLogName, "a")

    fLog.write(f"\n***********************************************************************************************\n\n")
    fLog.write(f"Place hydrogen for for {res}, {res.id[1]} of chain: {res.parent} \n")
    fLog.flush()
    
    structure = res.parent.parent.parent

    aboveSp3 = res['CB']

    if(res.resname == 'SER'): 
        sp3 = res['OG']
    else: 
        sp3 = res['OG1']
##################################REPEAT#####################################################
    customList = stp.get_knownDonorAcceptorListWRTOneAtom(structure,sp3, aaType = 'DONOR_ACCEPTOR')

    if(not customList):
        fLog.write(f"NO KNOWN CLOSE ATOMS FOUND for {res}, {res.id[1]} and chain: {res.parent}\n")
        fLog.flush()
        updatedLastSerial = lastSerial
        aatCoords = []
        fLog.close()
        return updatedLastSerial, aatCoords

    ##Get a list of all close Atoms (which are donor, acceptor, and both) for the given hvy atom
    allCloseAtoms = cats.get_allCloseAtomInfoForOneAtom(sp3, customList, debug =0)

    
    if(np.shape(allCloseAtoms)[0] == 1):
        fLog.write(f"NO CLOSE ATOMS FOUND for {res}, {res.id[1]} of chain: {res.parent}\n")
        fLog.flush()
        #print(f"NO CLOSE ATOMS FOUND for {res}, {res.id[1]} of chain: {res.parent}")
        updatedLastSerial = lastSerial
        aatCoords = []
        fLog.close()
        return updatedLastSerial, aatCoords

    aatInOriginal, aatInOriginalAppended, coordAngEnergy = computeAAtCoordsSp3_energy(sp3, aboveSp3, allCloseAtoms)
    aatCoords = aatInOriginal[-1]
    
    if(debug == True):
        print(f"Picking coord allAATInOriginal outside: {aatInOriginal[-1]}")
        checkBondAngForAllAAt(sp3.coord, aboveSp3.coord, aatCoords[0], aatCoords[1], aatCoords[2])

    names = ['HG', 'LP3', 'LP4'] if(res.resname == 'SER') else ['HG1', 'LP3', 'LP4']
    elements = ['H','LP','LP']
    
    for i in range(len(names)):

        res.add(Bio.PDB.Atom.Atom(name=names[i], coord=aatCoords[i], bfactor=0., occupancy=1., altloc=' ', fullname=names[i], serial_number=lastSerial+i,element=elements[i]))
    
    fLog.write(f"Side chain hydrogen placed, for {res}, added {names}, with s.no now at: {lastSerial+i} !\n")
    fLog.write(f"\n***********************************************************************************************\n\n")
    fLog.flush()

    stp.detectClashWrtResidue(structure,  res, withinStructClashDist = mc.btwResClashDist)
    updatedLastSerial = lastSerial + i

    fLog.close()
    return updatedLastSerial, aatCoords

####################################################################################################################

def placeHydrogens_LYS(res, lastSerial, method = 'energy', debug =0):
    '''
    objective: placing the hydrogen and Lone pairs for Lysine

        Input: -res: residue(LYS) that requires hydrogen placement
               -lastSerial: the serial number of the last atom in the structure

        Output: -updatedLastSerial: updated serial number of the added atom
                -hhCoords : three hydrogen coordinates attached to NZ

    '''

    fLogName = stp.get_logFileName(res.parent.parent.parent.id)
    fLog = open(fLogName, "a")
    fLog.write(f"\n***********************************************************************************************\n\n")
    fLog.write(f"Place hydrogen for for {res} of chain: {res.parent} \n")
    fLog.flush()

    aboveSp3 = res['CE']
    sp3 = res['NZ']
    structure = res.parent.parent.parent
    ###Get only the known atoms for generating close atoms!!!
    customList = stp.get_knownDonorAcceptorListWRTOneAtom(structure,sp3, aaType = 'DONOR_ACCEPTOR')

    if(not customList):
        fLog.write(f"NO KNOWN CLOSE ATOMS FOUND for {res}, {res.id[1]} of chain: {res.parent}\n")
        fLog.flush()
        hCoords = []
        updatedLastSerial = lastSerial
        fLog.close()
        return updatedLastSerial, hCoords

    ##Get a list of all close Atoms (which are donor and  acceptor) for the given hvy atom
    allCloseAtoms = cats.get_allCloseAtomInfoForOneAtom(sp3, customList, debug = 0)

    if(np.shape(allCloseAtoms)[0] == 1):
        fLog.write(f"NO CLOSE ATOMS FOUND for {res}, {res.id[1]} of chain: {res.parent}\n")
        fLog.flush()
        hCoords = []
        updatedLastSerial = lastSerial
        fLog.close()
        return updatedLastSerial, hCoords   
    #optimize and relevant coordinates of the hydrogens, associates angle and energy!!
    aatInOriginal, aatInOriginalAppended, coordAngEnergy = computeAAtCoordsSp3_energy(sp3, aboveSp3, allCloseAtoms)
    hCoords = aatInOriginal[-1]

    names = ['HZ1', 'HZ2', 'HZ3']
    
    #add the atoms to the residue
    for i in range(len(names)):
        res.add(Bio.PDB.Atom.Atom(name=names[i], coord=hCoords[i], bfactor=0., occupancy=1., altloc=' ', fullname=names[i], serial_number=lastSerial+i,element='H'))

    fLog.write(f"Side chain hydrogen placed, for {res} with chain:{res.parent}, added {len(names)} Hs, with s.no now at: {lastSerial+i} !\n")
    fLog.write(f"\n***********************************************************************************************\n\n")
    fLog.flush()
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
            stp.append2log(f"Not SP3/insufficient Atoms: {res.resname} with ID: {res.id[1]} and chain: {res.parent}\n" )
            pass
            
