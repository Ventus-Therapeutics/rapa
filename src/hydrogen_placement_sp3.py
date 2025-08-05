"""
This module contains computations related to placing the hydrogen that is
bonded to sp3 atoms.

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
import sys
import Bio
import numpy as np
from Bio.PDB import *


import my_residue_atom as mra
import setup_protein as stp
import close_atoms as cats
import my_math as mm
import global_constants as gc


def check_bond_angle_for_all_connected_atoms(sp3,abvSP3, sp3Coord, abvSP3Coord, aat0, aat1, aat2):

    """
        Objective: Verify the bond angle with all attached atoms
        Input: -sp3: sp3 atom
              -abvSP3: atom connected with sp3 that is not hydrogen or LP
              -sp3Coord: sp3 atom coord
              -abvSP3Coord: atom connected with sp3 that is not hydrogen or LP coord

              -aat0, aat1, aat2: Hyd/LP atoms
              
        Output: Prints out the angle between above SP3 atom-SP3 atom-hyd/LP atom

    """
    sp3Vec = Vector(sp3Coord)
    abvSP3Vec = Vector(abvSP3Coord)
    aat0Vec =Vector(aat0)
    aat1Vec =Vector(aat1)
    aat2Vec =Vector(aat2)
    if gc.debug:
        print(f" Checking for sp3: {sp3} of residue: {sp3.parent} of chain: {sp3.parent.parent}\n \
               aboveSP3 is: {abvSP3} of residue: {sp3.parent} of chain: {sp3.parent.parent}\n \
               sp3 angle should be approximately 109.5 degrees.\n \
               Calc angle:aboveSP3-SP3-Hcoord0 {calc_angle(abvSP3Vec, sp3Vec, aat0Vec)*180/3.14}, \n \
               Calc angle:aboveSP3-SP3-Hcoord1/LP1 {calc_angle(abvSP3Vec, sp3Vec, aat1Vec)*180/3.14},\n \
               Calc angle:aboveSP3-SP3-Hcoord2/LP2 {calc_angle(abvSP3Vec, sp3Vec, aat2Vec)*180/3.14}, \n \
    ")


def compute_initial_position_for_atoms_connected_to_sp3(sp3Coord, surrCoord, bondLength):
   
    """
    Objective: Calculate the initially attached atom/lone pair coordinate value. This
    may not be the actual value as it will be transformed into another frame and
    rotated to find minimum distance sum btw acceptors &  h-donors.
    Note-Adjustment to the angle of 109.5 is done at a later point

    Input:
         sp3Coord: coordinate of sp3 atom
         surrCoord: the surrounding atom or the closest atom wrt sp3 atom
         bondLength: bond length between sp3 atom and hydrogen/LP atom = 1 A

    Output:
         aat0PosVec: compute position vector of the first atom (Hyd/LP) connected with sp3 atom.
                    Note-Adjustment to the angle of 109.5 is done at a late point

    """
    #Find the direction
    sp3AAt0 = bondLength*mm.get_unit_vector(surrCoord-sp3Coord)
    #Find the position vector
    aat0PosVec = sp3AAt0 + sp3Coord

    return aat0PosVec


def create_prime_frame_of_reference(sp3Coord, aboveSp3Coord, aat0Coord):
    
    """
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

      
    """


    zHatPrime = mm.get_unit_vector(aboveSp3Coord - sp3Coord)
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

    xHatPrime = mm.get_unit_vector(xPrime)
    yHatPrime = np.cross(zHatPrime, xHatPrime)

    if gc.debug:
        print(f"New frame of reference has unit vectors: {xHatPrime}, {yHatPrime}, {zHatPrime}!")

    return xHatPrime, yHatPrime, zHatPrime


def setup_transformation_matrix( xHatPrime, yHatPrime, zHatPrime):
    
    """
    Set up the transform matrix 
    """

    return np.array([xHatPrime, yHatPrime, zHatPrime])


def move_coords_to_prime_reference_frame(shiftByCoord, transformMatrix, allCoordsToMove):

    """ 
        objective: Move the desired coordinates to the prime(new) frame of reference from original frame of reference
        Input: 
            -shiftByCoord: shift or translate by coordinates
            -transformMatrix: Transformation matrix for rotation(x prime, y prime, z prime)
            -allCoordsToMove: The coordinates to translate and rotate

        Output: transformedCoords: Newly transformed( translated/rotated) coordinates
    """
 
    ##First translate then rotate!!
    translatedCoords = mm.translate_coord(shiftByCoord, allCoordsToMove)
    ##A matrix vector multiplication to get transformed coordinates
    if(len(translatedCoords.shape)== 1):
        transformedCoords = np.einsum('ij,kj->ki', transformMatrix, [translatedCoords] )
    else:
        transformedCoords = np.einsum('ij,kj->ki', transformMatrix, translatedCoords ) 

    return transformedCoords



def move_coords_to_original_reference_frame(shiftCoord, transformMatrix, moveCoordsBack):

    """ 
    
    Objective: Move the desired coordinates to the original frame of reference from new frame of reference 
    Input: -shiftCoord/sp3Coord: Coordinates to translate or shift by (coordinate of sp3 atoms)
           -transformMatrix: Transform or Rotation matrix to get back to original frame of ref(xprime,yprime,zprime)
           -moveCoordsBack: Coordinates to move
    Output:-hBack: original/Transformed to original coordinate of the moveCoordsBack 

    """
    hBack = np.einsum("ij,kj->ki", np.linalg.inv(transformMatrix), moveCoordsBack) + shiftCoord    
    return hBack


def get_all_connected_atoms(aat0CoordPrime):

    
    """
        Objective: Get the other two attached atom coordinates given one attached atom coordinates 
                   in the new frame by rotating
        Input: -aat0CoordPrime: coordinate of the first attached atom of the sp3 atom
        Output: -aatPrime: array of the three attached atom to the sp3 atom that is 109.5 degrees apart
    """
    #Three atoms in plane at 120 degrees each
    rotInPlane = np.array((mm.get_rotation_matrix(120),mm.get_rotation_matrix(240)))

    aat1CoordPrime, aat2CoordPrime = np.einsum('ijk,k->ij',rotInPlane, aat0CoordPrime)
    aatPrime = np.array((aat0CoordPrime, aat1CoordPrime, aat2CoordPrime))
    
    return aatPrime 

    

def get_energy_of_all_close_atoms_for_hydrogen_lonepairs_connected_to_sp3(sp3, aatPrime, allCloseAtoms, ang,
                                                                          pre_cal_acceptor_donor_info=None):

    """
        objective: energy value of all close atoms for hydrogen and lone pairs
                    To find energy value of all close atoms for Hyd-0/Hyd-1/Hyd-2  is Sp3.parent ==LYS
                    or  Hyd-0/LP-1/LP-2 if sp3.parent == SER/THR

        input:-sp3: Sp3  atom
              -aatPrime: active atom prime coordinates (Hyd/lone pair)
              -allCloseAtoms: All close atoms
              -ang: angle at which energy values are being computed

        output: -aat0Energy, aat1Energy, aat2Energy: energies wrt  Hyd-0/Hyd-1/Hyd-2 or Hyd-0/LP-1/LP-
                -enSum: sum of the energies 
    """
    ##energy summation
    enSum = 0

    energyInteraction = []
    energyInteraction.append(['angle', 'sp3', 'sp3.parent', 'closeAt', 'closeAt.parent', 'aat0Energy, aat1Energy', 'aat2Energy', 'enSum'])
    ####################################################################################################
    for i in range(1, np.shape(allCloseAtoms)[0]):
        ##close atoms for which energy needs to be computed 
        closeAt = allCloseAtoms[i][0]
        myCloseAt = mra.my_atom(allCloseAtoms[i][0])

        if(myCloseAt.get_behavior().abbrev == 'ac'):
            ##Assume the first to be the Hydrogen atom (the others can be either two hydrogen-LYS or two lone pairs-SER/THR)
            aat0Energy, aat0enSum = cats.compute_energy_as_donor(closeAt, aatPrime[0], sp3, attractive=1, atype = 'SP3', pre_cal_donor_info=pre_cal_acceptor_donor_info['sp3'])

            if(sp3.parent.resname in ['LYS', 'LYN']):
                #For the case of two hydrogen:
                aat1Energy, aat1enSum = cats.compute_energy_as_donor(closeAt, aatPrime[1], sp3,  attractive=1, atype = 'SP3', pre_cal_donor_info=pre_cal_acceptor_donor_info['sp3'])
                aat2Energy, aat2enSum = cats.compute_energy_as_donor(closeAt, aatPrime[2], sp3,  attractive=1, atype = 'SP3', pre_cal_donor_info=pre_cal_acceptor_donor_info['sp3'])

            else:
                aat1Energy, aat1enSum = cats.compute_energy_as_acceptor(sp3, Vector(aatPrime[1]), closeAt, attractive = 0, atype = 'SP3',  pre_cal_acceptor_info=pre_cal_acceptor_donor_info['sp3'])
                aat2Energy, aat2enSum = cats.compute_energy_as_acceptor(sp3, Vector(aatPrime[2]), closeAt, attractive = 0, atype = 'SP3',  pre_cal_acceptor_info=pre_cal_acceptor_donor_info['sp3'])
                
            enSum = enSum + aat0enSum + aat1enSum + aat2enSum


        if(myCloseAt.get_behavior().abbrev == 'do'):
            ##Assume the first to be the hydrogen atom (the others can be either two hydrogen-LYS or two lone pairs-SER/THR)
            aat0Energy, aat0enSum = cats.compute_energy_as_donor(closeAt, aatPrime[0], sp3, attractive=0, atype= 'SP3', pre_cal_donor_info=pre_cal_acceptor_donor_info['sp3'])

            if(sp3.parent.resname in ['LYS', 'LYN']):
                #For the case of two hydrogen:
                aat1Energy, aat1enSum = cats.compute_energy_as_donor(closeAt, aatPrime[1], sp3, attractive =0,
                                                                     atype = 'SP3', pre_cal_donor_info=pre_cal_acceptor_donor_info['sp3'])
                aat2Energy, aat2enSum = cats.compute_energy_as_donor(closeAt, aatPrime[2], sp3, attractive =0, atype = 'SP3', pre_cal_donor_info=pre_cal_acceptor_donor_info['sp3'])

            else:
                aat1Energy, aat1enSum = cats.compute_energy_as_acceptor(sp3, Vector(aatPrime[1]), closeAt, attractive = 1, atype = 'SP3', pre_cal_acceptor_info=pre_cal_acceptor_donor_info['sp3'])
                aat2Energy, aat2enSum = cats.compute_energy_as_acceptor(sp3, Vector(aatPrime[2]), closeAt, attractive = 1, atype = 'SP3', pre_cal_acceptor_info=pre_cal_acceptor_donor_info['sp3'])

            enSum = enSum + aat0enSum + aat1enSum + aat2enSum

        energyInteraction.append([ang, sp3, sp3.parent, closeAt, closeAt.parent, aat0Energy,aat1Energy, aat2Energy, enSum])


    return aat0Energy, aat1Energy, aat2Energy, enSum
#

def optimize_connected_atoms_by_rotation_in_plane(sp3, aboveSp3, aatPrime,allCloseAtoms):

    """
        objective: Rotate in plane the three attached atom coordinates and 
         find sum of h-bonding energies  
        
        input:-sp3:sp3 atom
              -aboveSp3:atom connected to sp3 (which is not hydrogen or lone pair)
              -aatPrime:prime coordinate of the attached active atoms
              -allCloseAtoms: all close atoms
        
        output: -S: array of angle and energy at that angle
                -aatPrimeMat: attached active atom prime coordinates at each angle 
    """
    
    ####Minimize distance by changing h0_prime, h1_prime, h2_prime####################
    sumEnergyInfo = [] 
    aatPrimeMat = []

    #end angle for LYS: 120, and end angle for SER/THR: 360
    endAng = gc.optRotAngle_LYS if(sp3.parent.resname in ['LYS', 'LYN']) else gc.optRotAngle
    
   #precomputing to reduce time
    pre_cal_acceptor_donor_info= {'sp3': cats.get_info_for_acceptorAt_donorAt(sp3)}

    for ind,ang in enumerate(range(0,endAng,10)):
        aatPrimeRot = np.einsum("ij,kj->ki", mm.get_rotation_matrix(ang), aatPrime)
        aatPrimeMat.append(aatPrimeRot)
        aat0Energy, aat1Energy, aat2Energy, enSum = get_energy_of_all_close_atoms_for_hydrogen_lonepairs_connected_to_sp3(sp3, aatPrimeMat[ind], allCloseAtoms, ang, pre_cal_acceptor_donor_info = pre_cal_acceptor_donor_info)
        sumEnergyInfo.append([ang, enSum])

    S = np.array(sumEnergyInfo)
    
    return S, aatPrimeMat

def compute_atoms_connected_to_sp3_in_prime_reference_frame(sp3, aboveSp3, allCloseAtoms, aat0CoordPrime):
     
    """
        objective: compute optimized( or minimum energy interaction) coordinate of the attached atoms(H0/H1/H2 or H0/LP1/LP2) in the new frame of reference using the energy method
        input:  -sp3: sp3 atom
                -aboveSp3: atom connected to the sp3 atom(not hydrogen or lone pairs)
                -allCloseAtoms: all relevant(knowns-donor/acceptor) closeAtoms that need to be explored
                -aat0CoordPrime: coordinate of first attached active atom in the prime frame of reference

        output: -aatCoordPrime: optimized coordinate value of the attached atoms
                -S: array of angle and energy at that angle
                -aatPrimePositionMat: all coordinate possibilities(rotated in the plane) of attached atom 

    """     
    

    #get all the attached/connected/bonded atom
    aatPrime = get_all_connected_atoms(aat0CoordPrime)

    #############################################################################
    #Find the coordinates for hydrogen and lone pair placement by rotating in plane and computing S i.e angle and energy computed at that angle, along with aatPrimePositionMat or attached atom coordinates
    S, aatPrimePositionMat  = optimize_connected_atoms_by_rotation_in_plane(sp3, aboveSp3, aatPrime, allCloseAtoms)

        
    #Now we will optimize and pick the  appropriate coordinate
    ##Check if all energy values are 0!!
    if(all(x == 0 for x in S[:,-1].flatten())):
      aatCoordPrime = aatPrimePositionMat[0] #first value is at 0 degrees, next at 10, then at 20, etc degrees
      if gc.log_file: print("Taking h-bonding possibility at angle 0, as all energy values=0")

      return aatCoordPrime, S, aatPrimePositionMat

    
    index =  S[:,-1].argsort(axis=0)[::-1] ##Descending order according to energy value

    ##arranging S from max to min of energy values
    S = S[index]
   
    #all the attached atom prime coordinates in an array and in order from max to min
    aatPrimePositionMat = np.array(aatPrimePositionMat)[index]
    #Pick up the coordinate at the end as it is arranged in descending order
    aatCoordPrime = aatPrimePositionMat[-1]
    minV = np.amin(S[:,-1]) #minimum energy value

    return aatCoordPrime, S, aatPrimePositionMat


def adjust_first_connected_atom_to_sp3_in_prime_reference_frame(aboveSp3CoordPrime, sp3CoordPrime, aat0CoordPrime):
    
     
    """
     objective: Modify the initial attached atom coordinate which may be 
     from 90-120 degrees to exactly 109 degrees as per SP3 requirement

    Input: -aboveSp3CoordPrime: atom(non hydrogen/ non-lone pair) connected to sp3 in prime frame
           -sp3CoordPrime: sp3 atom coord in prime frame
           -aat0CoordPrime: initial attached atom coordinates in prime frame 

    Output: aat0CoordPrimeMod: modified coordinates of the attached active atom0(Hydrogen) 

    """



    V0 = Vector(aboveSp3CoordPrime)
    V1 = Vector(sp3CoordPrime )
    V2 = Vector(aat0CoordPrime)
    adjustTheta = gc.sp3Angle-(calc_angle(V0, V1, V2)*180/np.pi)
    aat0CoordPrimeMod = np.einsum('ij,j->i',mm.get_rotation_matrix_about_Yaxis(adjustTheta), aat0CoordPrime)
    
    return aat0CoordPrimeMod

def get_close_atom_coords(closeAtoms):

    
    """
    objective: Get close atom coords 
    input: 
        -closeAtoms: list of close atoms in format: [atomClose, atomClose.get_parent().id[1],atomClose.get_parent().resname, aClose.get_behavior().abbrev, dist], example: [<Atom ND1>, 25, 'HIE', 'ac', 2.663196],
    output: 
        -closeAtomCoords: list of the coordinates of each close atom
    """
    numCloseAtoms =  np.shape(closeAtoms)[0] 
    closeAtomCoords =[]
    for i in range(1,numCloseAtoms):
        closeAtomCoords.append(closeAtoms[i][0].coord)
    return closeAtomCoords


def set_coords_in_prime_reference_frame(sp3, aboveSp3, allCloseAtoms, sp3CoordPrime, aboveSp3CoordPrime, closeAtomCoordsPrime):

    """ 
        objective: To set sp3, connected to sp3 and close atoms coordinate in the prime frame of reference
        Input:-sp3: sp3 atom
              -aboveSp3: atom(non hydrogen/lone pair) connected to sp3 atom
              -allCloseAtoms: all relevant close atoms
              -sp3CoordPrime: sp3 atom coordinate in prime frame of reference
              -aboveSp3CoordPrime: atom(non hydrogen/lone pair) connected to sp3 atom in prime frame of reference
              -closeAtomCoordsPrime: all close atom coordinates in prime frame of reference 

       Output: setting the given atom's coordinates in the prime frame of reference

    """

    sp3.set_coord(sp3CoordPrime)
    aboveSp3.set_coord(aboveSp3CoordPrime)

    if(allCloseAtoms !=[]):
        if(np.shape(closeAtomCoordsPrime)[0]== 1 ):
            allCloseAtoms[1][0].set_coord(np.array([closeAtomCoordsPrime[0][0], closeAtomCoordsPrime[0][1], closeAtomCoordsPrime[0][2]]))
        else:
            for i in range(1, np.shape(allCloseAtoms)[0]):
                allCloseAtoms[i][0].set_coord(closeAtomCoordsPrime[i-1])

def set_coords_in_original_reference_frame(sp3, aboveSp3, allCloseAtoms, sp3Coord, aboveSp3Coord, allCloseAtomCoords):
       
    """
        objective: To set sp3, connected to sp3 and close atoms coordinate in the original frame of reference
        Input:-sp3: sp3 atom
              -aboveSp3: atom(non hydrogen/lone pair) connected to sp3 atom
              -allCloseAtoms: all relevant close atoms
              -sp3CoordPrime: sp3 atom coordinate in original frame of reference
              -aboveSp3CoordPrime: atom(non hydrogen/lone pair) connected to sp3 atom in original frame of reference
              -closeAtomCoordsPrime: all close atom coordinates in original frame of reference 

       Output: setting the given atom's coordinates in the original frame of reference
    """

    sp3.set_coord(sp3Coord)
    aboveSp3.set_coord(aboveSp3Coord)

    if(allCloseAtoms !=[]):
        moveCloseAtoms = np.array(allCloseAtomCoords)
        if(len(moveCloseAtoms.shape)== 1 ):
            allCloseAtoms[1][0].set_coords(moveCloseAtoms)
        else:
            for i in range(1, np.shape(allCloseAtoms)[0]):
                allCloseAtoms[i][0].set_coord(moveCloseAtoms[i-1])


def compute_connected_atoms_to_sp3(sp3, aboveSp3, allCloseAtoms):

     """
        Objective: To compute optimized attached atom coordinates(Hydrogen and lone pairs) for the SP3 atom

        Input: -sp3: SP3 atom
               -aboveSP3: atom (non hydrogen/lone pair) connected to sp3 atom 
               -allCloseAtoms: the relevant close atoms around the sp3 atom

        Output: -allAAtInOriginal:all the potential attached atom coordinates in original frame of reference.
                                  The last item in the list is the optimized coordinate for the three 
                                  positions(h0/h1/h2 or h0/LP1/ LP2). This is also easier to read/access
                -allAAt:all the potential attached atom coordinates in original frame of reference.The last three
                                  coordinates in the list is the optimized coordinate picked. This is differently
                                  arranged than allAAtInOriginal, incase we want to pick one coordinate at a time

                -coordAngEnergy: optimized coordinate and its associated angle (as its rotated to find optimum value)
                                 and its associated energy (SrevStored). It has shape: np.shape(coordAngEnergy)= (1, 2).
                                 -coordAngEnergy[0][0]: optimized coordinate value, np.shape(coordAngEnergy[0][0]) =(3,3)
                                 -coordAngEnergy[0][1]: associated angle & energy, np.shape(coordAngEnergy[0][1])= (2,1)
     """    


        


     bondLen = 1
     sp3Coord = sp3.coord
     aboveSp3Coord = aboveSp3.coord

     aat0Coord = compute_initial_position_for_atoms_connected_to_sp3(sp3Coord, allCloseAtoms[1][0].coord, bondLen)

#    ##Create the new frame of reference
     xHatPrime, yHatPrime, zHatPrime = create_prime_frame_of_reference(sp3Coord, aboveSp3Coord, aat0Coord)
#    ##Create the transformation matrix
     transformMatrix = setup_transformation_matrix( xHatPrime, yHatPrime, zHatPrime)

     moveCoords = np.array((sp3Coord, aboveSp3Coord, aat0Coord)) 
     primeCoords = move_coords_to_prime_reference_frame(sp3Coord, transformMatrix, moveCoords)
     sp3CoordPrime = primeCoords[0]
     aboveSp3CoordPrime = primeCoords[1]
     aat0CoordPrime = primeCoords[2]

#################### Move to new frame of reference###########################

     if(allCloseAtoms):
        closeAtomCoords = get_close_atom_coords(allCloseAtoms)
        closeAtomCoordsPrime = move_coords_to_prime_reference_frame(sp3Coord, transformMatrix, closeAtomCoords)

     set_coords_in_prime_reference_frame(sp3, aboveSp3, allCloseAtoms, sp3CoordPrime, aboveSp3CoordPrime, closeAtomCoordsPrime)
##########################In the new frame ####################################################################
     aat0CoordPrimeMod = adjust_first_connected_atom_to_sp3_in_prime_reference_frame(aboveSp3.coord, sp3.coord, aat0CoordPrime)
     aatPrimeDistMin, S, aatPrimeMat = compute_atoms_connected_to_sp3_in_prime_reference_frame(sp3, aboveSp3, allCloseAtoms, aat0CoordPrimeMod)

#########################End the new frame of reference####################################################################
     set_coords_in_original_reference_frame(sp3, aboveSp3, allCloseAtoms, sp3Coord, aboveSp3Coord, closeAtomCoords)
#################### Get back to original frame of reference ###########################
    
     numRows = np.shape(aatPrimeMat)[0]*np.shape(aatPrimeMat)[1]
     
     allAAt = move_coords_to_original_reference_frame(sp3Coord, transformMatrix, np.reshape(aatPrimeMat,(numRows,3)))
     numAAt = int(np.shape(allAAt)[0]/3)
     allAAtInOriginal = np.reshape(allAAt,(numAAt,3,3))
   
     coordAngEnergy = []
     Srev = S[::-1]
    
######Check if the optimized coordinate picked clashes with any surrounding atoms, if yes: pick next best option#######################
     for i,AAtInOriginal in reversed(list(enumerate(allAAtInOriginal))):
         #since LYS has 3 attached hydrogen that need to be checked, else only one hydrogen is in SER/THR.
         #so all the hydrogen needs to be checked for  the clash. Lone pairs are only pseudo atoms-so we do not check that

         hMax=3 if sp3.parent.resname in ['LYS', 'LYN'] else 1
        

         for j in range(hMax):
             coord2check  = AAtInOriginal[j]
             if gc.log_file:
                print(f"Need to look for coord so H does not clash. max hyd={hMax}. Currently at iteration:i:{i}")

             residue = sp3.parent
             structure = residue.parent.parent.parent

             resSearchList  = stp.get_all_residues(structure, donotIncludeRes = residue)

             searchListAtoms = []
             for res in resSearchList:
                    searchListAtoms.append(stp.create_list_of_atoms_of_residue_without_lonepair(res))


             searchListAtoms =  [atom for atomlist in searchListAtoms for atom in atomlist]
        #    ###use the nbd search and create an object ns.
             ns = Bio.PDB.NeighborSearch(searchListAtoms)
             potClashAtom = ns.search(coord2check,1.5)

             if(potClashAtom):
                if gc.log_file:
                    print(f"potential clash atom = {potClashAtom}")

                for pcAtom in potClashAtom:
                    if gc.log_file:
                        print(f"orignal atom: pot Hyd, its parent: {sp3.parent}, pcAtom:{pcAtom} and parent: {pcAtom.parent} and distance is: {np.linalg.norm(coord2check-np.float32(pcAtom.coord))}")
                continue
             else: 
                if(j != hMax-1):
                    continue
                else:
                    SrevStored=Srev[j].reshape(2,1) #first in the list is angle, the other is energy
                    coordAngEnergy.append( [AAtInOriginal, SrevStored])
                    allAAt = np.append(allAAt, AAtInOriginal, axis=0)
                    allAAtInOriginal = np.append(allAAtInOriginal, [AAtInOriginal], axis =0)

                    return allAAtInOriginal, allAAt, coordAngEnergy



def place_hydrogens_lonepairs_SER_THR(res, lastSerial):


    """
        objective: placing the hydrogen and Lone pairs for Serine

        Input: -res: residue(SER/THR) that requires hydrogen and lone pair placement 
               -lastSerial: the serial number of the last atom in the structure

        Output: -updatedLastSerial: updated serial number of the added atom
                -aatCoords : attached atom (Hydrogen and lone pairs) coordinates
    """
    if gc.log_file:
        print(f"\n***********************************************************************************************\n")
        print(f"Place hydrogen for for {res}, {res.id[1]} of chain: {res.parent} ")
    
    structure = res.parent.parent.parent

    aboveSp3 = res['CB']

    if(res.resname == 'SER'): 
        sp3 = res['OG']
    else: 
        sp3 = res['OG1']

    customList = stp.get_known_donor_acceptor_list_for_one_atom(structure,sp3, aaType = 'DONOR_ACCEPTOR')

    if(not customList):
        if gc.log_file:
            print(f"NO known close atoms found for {res}, {res.id[1]} and chain: {res.parent}")
        updatedLastSerial = lastSerial
        aatCoords = []
        return updatedLastSerial, aatCoords

    ##Get a list of all close Atoms (which are donor, acceptor, and both) for the given heavy atom
    allCloseAtoms = cats.get_all_close_atom_info_for_one_atom(sp3, customList)

    
    if(np.shape(allCloseAtoms)[0] == 1):
        if gc.log_file:
            print(f"No close atoms found for {res}, {res.id[1]} of chain: {res.parent}")

        updatedLastSerial = lastSerial
        aatCoords = []
        return updatedLastSerial, aatCoords


    aatInOriginal, aatInOriginalAppended, coordAngEnergy = compute_connected_atoms_to_sp3(sp3, aboveSp3, allCloseAtoms)
    aatCoords = aatInOriginal[-1]

    names = ['HG', 'LP3', 'LP4'] if(res.resname == 'SER') else ['HG1', 'LP3', 'LP4']
    elements = ['H','LP','LP']

    #add atoms to reside:
    for i in range(len(names)):

        res.add(Bio.PDB.Atom.Atom(name=names[i], coord=aatCoords[i], bfactor=0., occupancy=1., altloc=' ', fullname=names[i], serial_number=lastSerial+i,element=elements[i]))
        if gc.log_file:
            print(f"Side chain hydrogen placed, for {res} with chain: {res.parent}, added {names}, with s.no now at: {lastSerial+i} !")
            print(f"\n***********************************************************************************************\n")

    updatedLastSerial = lastSerial + i

    return updatedLastSerial, aatCoords


def place_hydrogens_lonepairs_LYS(res, lastSerial):
     
    """
    objective: placing the hydrogen and Lone pairs for Lysine

        Input: 
               -res: residue(LYS) that requires hydrogen placement
               -lastSerial: the serial number of the last atom in the structure

        Output: 
                -updatedLastSerial: updated serial number of the added atom
                -hhCoords : three hydrogen coordinates attached to NZ
    """
    
    
    if gc.log_file:
        print(f"\n***********************************************************************************************\n")
        print(f"Place hydrogen for for {res} of chain: {res.parent}")

    aboveSp3 = res['CE']
    sp3 = res['NZ']
    structure = res.parent.parent.parent
    ###Get only the known atoms for generating close atoms
    customList = stp.get_known_donor_acceptor_list_for_one_atom(structure,sp3, aaType = 'DONOR_ACCEPTOR')



    if(not customList):
        if gc.log_file:
            print(f"No known close atoms found for {res}, {res.id[1]} of chain: {res.parent}")

        hCoords = []
        updatedLastSerial = lastSerial
        return updatedLastSerial, hCoords

    ##Get a list of all close Atoms (which are donor and  acceptor) for the given heavy atom
    allCloseAtoms = cats.get_all_close_atom_info_for_one_atom(sp3, customList)

    if(np.shape(allCloseAtoms)[0] == 1):
        if gc.log_file:
            print(f"No close atoms found for {res}, {res.id[1]} of chain: {res.parent}")

        hCoords = []
        updatedLastSerial = lastSerial
        return updatedLastSerial, hCoords   

    #optimize and relevant coordinates of the hydrogen, associated angle and energy
    aatInOriginal, aatInOriginalAppended, coordAngEnergy = compute_connected_atoms_to_sp3(sp3, aboveSp3, allCloseAtoms)
    hCoords = aatInOriginal[-1]

    names = ['HZ1', 'HZ2', 'HZ3']
    
    #add the atoms to the residue
    for i in range(len(names)):
        res.add(Bio.PDB.Atom.Atom(name=names[i], coord=hCoords[i], bfactor=0., occupancy=1., altloc=' ', fullname=names[i], serial_number=lastSerial+i,element='H'))
    
    if gc.log_file:
        print(f"Side chain hydrogen placed, for {res} with chain:{res.parent}, added {len(names)} Hs, with s.no now at: {lastSerial+i} !")
        print(f"\n***********************************************************************************************\n")

    updatedLastSerial = lastSerial+i
   
    return updatedLastSerial, hCoords

