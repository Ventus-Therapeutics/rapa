
from sympy import *
import sys
import Bio 
import numpy as np
import code
import math
import os

from collections import namedtuple
from Bio.PDB import *

import my_math as mm
import my_residue_atom as mra
import my_constants as mc
import setup_protein as stp
import close_atoms as cats


def compute_plane_parameters(heavyAtoms, debug=0):
    ''' objective: computing the coefficients of the plane
        Typically plane is defined as: ax+by+cz =d.
        Here for easy computing we divide entire
        equation with d and solve for: Ax+By+Cz =1.
        -input:heavyAtoms:[[hvyAtm1_x, hvyAtm1_y, hvyAt1_z], [hvyAtm2_x, hvyAtm2_y, hvyAt2_z], [hvyAtm3_x, hvyAtm3_y, hvyAt3_z]]
        -output:coefficient of the plane equation
    '''
    b = np.array([1.0, 1.0, 1.0])        
    sol = np.linalg.solve(heavyAtoms,b)
    if(debug == 1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name,f"The heavy atoms are: {heavyAtoms}, with b={b}, and planar coefficients: {sol}\n", debug=0) 
    return sol 


def solve_system(heavyAtoms, coeffs, theta, debug = 0):
    ''' objective: To find hydrogen coordinate for the SP2 atom attached to hydrogen. 
        3 variables:coordinates of hydrogen atom: x,y,z
        3 conditions:1)bond length=1, 2)Planar 3) angle btw appropriate vectors=120
       Hydrogen will sit on this plane therefore satisfying,
       A*x0+B*y_0+C*z_0=1.  We then subtract A*x1+B*y1+C*z1 =1
       So we get: A*(x0-x1)+B*(y0-y1)+C*(z0-z1) = 0.
       
       Redefining: x =x0-x1, y=y0-y1, z=z0-z1. 
       We will look for (x0,y0,z0) at the end by adding N_bb coords/(heavyAtom[1]).
       
       Angle btw Hydrogen, Heavy1(Nitrogen) & Heavy2(CA) = 120 degrees
        r5 dot r6 = mod(r5)*mod(r6)*cos(120) 
       mod(r5)=1 as bond length of N-H=1 
       r5 = vector from bb Nitrogen--> Hydrogen (Heavy 1 to H)
       r6 = vector from Nitrogen bb  --> CA (Heavy 1 to Heavy 2)
       First compute r6
       Note: In above condition only Heavy-1, Heavy 2 are involved.
       Heavy0 (heavy connected to prev res) is NOT involved.
    
       input:  -heavyatoms: the coordinates of heavy atoms involved in the system
                            1)directly connected to hyd 2)atom connected to heavy1
                            3)third atom in plane typically the other atom connected to heavy1

               -coeffs: coefficients of the plane in which the heavy atoms lie
               -theta: angle btw two vectors: 120 degrees (as it is SP2) case
               -printDetails: get more detailed output
        output: hydrogen coordinates
       '''
      
    hv1_hv2 = heavyAtoms[2] - heavyAtoms[1] ###r6

    hv1_hv2_mod = np.sqrt(hv1_hv2.dot(hv1_hv2))

    
    x, y, z = symbols('x, y, z')
    eq1 = Eq(coeffs[0]*x+ coeffs[1]*y+coeffs[2]*z,0)  ##ONLY true if planar
    eq2 = Eq(hv1_hv2[0]*x +hv1_hv2[1]*y+hv1_hv2[2]*z,hv1_hv2_mod*np.cos(math.radians(theta) )) #r6.r5 =cos(120)##Depends on the angle between Hydrogen and the heavy atom
    eq3 = Eq(x**2+y**2+z**2,1) ##This is always going to be true!!
    sol_exact = solve([eq1, eq2, eq3], [x,y,z])
    

    HCoord = sol_exact + heavyAtoms[1]
    
    if(debug==1):
       stp.append_to_debug( __name__, sys._getframe().f_code.co_name, f"\n############################################## \n Exactly solving these equations simultaneously\n \
        Eqn1: {coeffs[0]} x + {coeffs[1]} y + {coeffs[2]} z = 0 \n \
        Eqn2: {hv1_hv2[0]} x + {hv1_hv2[1]} y + {hv1_hv2[2]} z = {hv1_hv2_mod}cos(120)\n \
        Eqn3: x^2 + y^2 + z^2 = 1\n \
\n############################################## \n \
\n############################################## \n \
    H coordinate is: {HCoord} \n \
    SOL in shifted frame: {sol_exact}\n \
    Check solutions, solution 1:\n \
    Eq1: {coeffs[0]*sol_exact[0][0]+ coeffs[1]*sol_exact[0][1] + coeffs[2]*sol_exact[0][2]} = 0?\n \
    Eq2: {hv1_hv2[0]*sol_exact[0][0]+ hv1_hv2[1]*sol_exact[0][1] + hv1_hv2[2]*sol_exact[0][2]} = {hv1_hv2_mod*np.cos(theta*np.pi/180)} ? \n \
    Eq3: {sol_exact[0][0]**2+ sol_exact[0][1]**2 + sol_exact[0][2]**2} = 1 ?\n \
    Check solutions, solution 2:\n \
    Eq1: {coeffs[0]*sol_exact[1][0]+ coeffs[1]*sol_exact[1][1] + coeffs[2]*sol_exact[1][2]} = 0?\n \
    Eq2: {hv1_hv2[0]*sol_exact[1][0]+ hv1_hv2[1]*sol_exact[1][1] + hv1_hv2[2]*sol_exact[1][2]} = {hv1_hv2_mod*np.cos(theta*np.pi/180)} ?\n \
    Eq3: {sol_exact[1][0]**2+ sol_exact[1][1]**2 + sol_exact[1][2]**2} = 1 ?\n \
    \n############################################## \n \
               ", debug=0)

    return HCoord
       
def eliminate_extraneous_solution(HCoord, vecHeavy, theta, debug=0):

    '''
        objective: Since there is a quadratic involved(bond length/distanc =1)-we get two solutions. Need to pick one.
         We get 2 solutions as Heavy1 atom attached to heavy2 atom can have atoms at 120degrees from two sides 
         on the plane. There is a need to eliminate the incorrect solution. Taking example of backbone here:
        Nitrogen makes 3 bonds at 120 degrees.
       Bond 1: Prev resiude Carbonyl carbon (Heavy 3). 
       Bond 2: Current Res Carbon Alpha . 
       Bond 3: Hydrogen on current res
       We get two solutions on the given plane. One: CA-N-C/H_computed and second: CA-N-H_computed
       We will elimate: CA-N-C/H as it is simply overlapping with carbon. Note this is happening as:
        our constraint involved only r5 dot r6 = mod r5 mod r6 * cos(120). We LEAVE OUT r4 (involving Heavy 3)! The solution could be a value on vector r4 or vector r5.
       r4 vector is from Carbonyl carbon (Heavy 3) to nitrogen(Heavy1): (C--->N)
        Since we leave out r4. We will compare the angle of Heavy-3, Heavy1, Hydrogen Coord with theta (what it should be).       If the difference is smaller: leave that solution- as that is suppose to be  the same coordinates as Heavy 3.
    
    input:-HCoord: the two hydrogen coordinates
          -vecHeavy:   the coordinates of heavy atoms involved in the system
                            1)directly connected to hyd 2)atom connected to heavy1
                            3)third atom in plane typically the other atom connected to heavy1
          -theta: 120 Degrees as we are looking at SP2 case
          -printDetails: Flag to print out more details
    output:
          -HCoordFinal: Final hydrogen coordinates picked!
    '''

    Vec_H1 = Vector(HCoord[0])
    Vec_H2 = Vector(HCoord[1])
##My formulation is wrt CA. So Now I am going to check the angle wrt to C (Heavy 3). Whatever solution is closer to theta will be used. Other will be dumped.
    
    ang_hv0_hv1_hd = np.zeros(np.shape(HCoord)[0])       
    ang_hv0_hv1_hd[0] = calc_angle(vecHeavy[0], vecHeavy[1], Vec_H1)*180/np.pi
    ang_hv0_hv1_hd[1] = calc_angle(vecHeavy[0], vecHeavy[1], Vec_H2)*180/np.pi
    
    ang_comparison = np.absolute(ang_hv0_hv1_hd-theta)
    min_diff_ind = np.where(ang_comparison == min(ang_comparison))[0][0]

    HCoordFinal = HCoord[min_diff_ind]
       
    if(debug ==1):
        stp.append_to_debug( __name__, sys._getframe().f_code.co_name, f"\n Diff wrt sol1: {ang_comparison[0]} vs sol2: {ang_comparison[1]}.\n Picking Coord IND {min_diff_ind} as its closest to theta, thus Hydrogen coord is: {HCoordFinal} \n", debug=0)

    return HCoordFinal


def compute_hydrogen_coords_sp2( heavyAtoms, clashOccurs, debug =0):
    ''' 
        heavy1: directly connected to HYD
        heavy2: atom connected to heavy1
        heavy0: Third atom in plane typically the other atom connected to heavy1
        Compute hydcoord for a sp2 case using 3 constraints:
        1. Bond length =1
        2. Sp2 is planar
        3. Angle btw heavy and unknown = 120 degrees
        I/P: 3 heavy atom coords
        O/P:  Hydrogen coords

    '''
   
    Vec_heavy0  = Vector(heavyAtoms[0])
    Vec_heavy1 = Vector(heavyAtoms[1])
    Vec_heavy2 = Vector(heavyAtoms[2])
    
    if clashOccurs == 1:
        theta = ((2*np.pi-calc_angle(Vec_heavy0,Vec_heavy1,Vec_heavy2))/2)*180/np.pi ##This should be approximately 120!!
    else:
        theta = mc.sp2Angle #120 degrees
    
    vecHeavy = np.array([Vec_heavy0, Vec_heavy1, Vec_heavy2])
    coeffs = compute_plane_parameters(heavyAtoms, debug=0)

    hCoord = solve_system(heavyAtoms, coeffs, theta, debug=debug)
    if clashOccurs == 1: 
        hCoordFinal = eliminate_extraneous_solution(hCoord, vecHeavy, theta, debug=debug)
    else:
        hCoordFinal = hCoord

    if(debug==1):
        stp.append_to_debug(__name__, sys._getframe().f_code.co_name, f' \n angle between sp2 atoms should be approximately 120, and it is: {theta}. Note, clash occurs is: {clashOccurs} and final hydrogen coordinates are: {hCoordFinal}\n', debug=0)

    return hCoordFinal


def add_lonepair_to_residue(res, LPname, lpCoords, countSerial, LPcoordsInfo, debug=0):
    '''
       objective: To add lone pair to a given residue 
       input:-res: residue to which the lone pair needs to be added
             -LPname: name of the lone pair to add
             -lpCoords: coords of the lone pair to add
             -countSerial: serial number of the last atom in the structure
             -LPcoordsInfo: Information of the lone pair coords: countSerial, number, residue, lone pair coords 
             -debug: flag to print additional details

       output:-countSerial: updated serial number for the last atom in the structure
              -LPcoordsInfo: Information regarding the lone pair coordinates including
                :countSerial, number, residue, lone pair coords 

    '''
    for i in range(len(LPname)):
        countSerial+=1 
        res.add(Atom.Atom(name = LPname[i], coord=lpCoords[i], bfactor=0., occupancy=1., altloc=' ', fullname=LPname[i], serial_number=countSerial, element='LP'))
        LPcoordsInfo.append([countSerial, i, res, lpCoords])

        if debug ==1: 
            stp.append_to_debug_less_detail( f"{countSerial}, {i}, {res.id}, {res.resname}, with LPname: {LPname[i]}, lone pair coordinates: {lpCoords[i]}\n  Lonepair placed, for {res}, chain:{res.parent} with LPname: {LPname[i]}, with s.no now at: {countSerial} !\n", debug=0 )
      
    return countSerial, LPcoordsInfo


def place_lonepair_on_backbone(structure, lastSerial, debug = 0):

    '''
        objective:Adding lonepair for all backbones in the amino-acid residues using
        current Oxygen coordinates, carbon-alpha coords, and carbonyl carbon coords.
        input: 
               -structure:the structure in consideration (who's all residues backbones need lonepairs)
               -modelID: the modelID in consideration (who's all residues backbones need lonepairs)
               -chainID: the chainID in consideration (who's all residues backbones need lonepairs)
               -lastSerial: the serial number of the last atom in the structure
               -printDetails: flag for printing more details
        output:
               -countSerial: updated serial number of the added atom
               -bbLPcoordsInfo: Additional info regarding the lone pair added to the backbone
                including: countSerial, number, residue, lone pair coords 
        '''
    countSerial = lastSerial
    bbLPcoordsInfo = []

    for model in structure.child_list:
        for chain in model.child_list: 
            count = 0
            for res in chain.child_list: 
                myRes = mra.my_residue(res)
               
                if(myRes.is_valid_amino_acid()==1):

                    resACE = False
                    count+=1
                    
                    #Mark if the first residue is an ACE
                    if(count==1 and myRes.is_valid_amino_acid()==1 ):
                        resACE = res.resname =='ACE'
                     
                    if(myRes.is_valid_amino_acid()==1 and resACE == False and res.resname != 'NME'): 
                    ##youshould be able to put LP on ACE if it is at the end!
                        heavy0 = res['CA'].coord
                        heavy1 = res['O'].coord
                        heavy2 = res['C'].coord
                        lpCoordBB = compute_hydrogen_coords_sp2( np.array([heavy0, heavy1, heavy2]),  clashOccurs=0, debug=debug) 
                        LPname = ['LP1', 'LP2'] 
                        countSerial, bbLPcoordsInfo = add_lonepair_to_residue(res, LPname, lpCoordBB, countSerial, bbLPcoordsInfo, debug=debug)
    return countSerial, bbLPcoordsInfo

def place_lonepair(res,lastSerial, hvys, LPnameAll, debug=0):
    ''' 
        Adding lonepair for a given residue
        input:-res: residue to which the lone pair is associated with
              -lastSerial: serial number for the last atom in the structure
              -hvys: All the heavy atoms associated with residue and the lone pair
              -LPnameAll: Name of all the lone pairs that are added
              -debug: print out additional details

        output:-countSerial: the latest serial number after adding lone pair atom
               -LPcoordsInfo:Information regarding the lone pair coordinates that are added, including:
               countSerial, number, residue, lone pair coords 
        '''
    
    countSerial = lastSerial
    LPcoordsInfo = []
    if(debug==1):
        stp.append_to_debug_less_detail(f"Now placing LPs: {LPnameAll} for {res} with chain: {res.parent}.\n", debug=0 )

    for i in range(np.shape(hvys)[0]):
            heavy0 = res[hvys[i][0]].coord
            heavy1 = res[hvys[i][1]].coord
            heavy2 = res[hvys[i][2]].coord

            lpCoord = compute_hydrogen_coords_sp2( np.array([heavy0, heavy1, heavy2]),  clashOccurs=0, debug=debug) 
            LPname = LPnameAll[i]
            countSerial, LPcoords = add_lonepair_to_residue(res, LPname, lpCoord, countSerial, LPcoordsInfo, debug=debug)
       
    return countSerial, LPcoordsInfo


def placeHydrogens_backbone(structure, lastSerial, debug = 0):
    '''
        objective: Adding hydrogens for all backbones in the amino-acid residues using
        current nitrogen coordinates, carbon-alpha coords, and 
        connected prev residues' carbonyl carbon coords.
        input:
           -structure:the structure in consideration (who's all residues backbones need hydrogen)
           -lastSerial: the serial number of the last atom in the structure
           -debug: flag for printing more details

        output:
            lastSerial+i: the updated serial number for the last atom in the structure
            bbHcoordsInfo: backbone hydrogen coords Information regarding the H-atom coordinates that are added, including:
               countSerial, number, residue, H-atom coords 
        '''
    if(debug==1):
        stp.start_debug_file( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debug_file_name(debug=0)
        fd = open(fDebugName, "a")

    for model in structure.child_list:
        for chain in model.child_list:
            
            nitrogenList = []
            carbonList = []
            carbonAlphaList =[]

            for atom in chain.get_atoms():
                a = mra.my_atom(atom)

                if(mra.my_residue(a.get_parent()).is_valid_amino_acid()==1): 
                    if(a.get_name() == 'N'):
                        nitrogenList.append(a)

                    if(a.get_name() == 'CA'):
                        carbonAlphaList.append(a)

                    if(a.get_name() == 'C'):
                        carbonList.append(a)

            stp.append_to_log(f"Collected atoms for chain: {chain} and now I will fix backbone H \n", debug=0)
            if(debug==1):
                fd.write(f"Collected atoms for chain: {chain} and now I will fix backbone  H \n")
                fd.flush()


            bbHcoordsInfo =[]
            firstResACE = 0
            if((not carbonList) or (not carbonAlphaList) or (not nitrogenList) ): continue
            #Mark if first res is ace
            if(carbonList[0].parent.resname == 'ACE'): firstResACE = 1

            iterOver = len(carbonList)
            for i in np.arange(1,iterOver,1):#If first res is ACE-new try
            ##When there is ACE at the start of the residue-we have mismatched length of 
               heavy0  = carbonList[i-1].coord#IF first res is NOT ACE

               if(firstResACE == 1):
                      res = nitrogenList[i-1].parent
                      heavy1 = nitrogenList[i-1].coord #N_bb = nitrogenList[i][-1]
                      heavy2 = carbonAlphaList[i-1].coord
               else: 
                    heavy1 = nitrogenList[i].coord #N_bb = nitrogenList[i][-1]
                    heavy2 = carbonAlphaList[i].coord
                    res = nitrogenList[i].parent

               hCoordBB = compute_hydrogen_coords_sp2( np.array([heavy0, heavy1, heavy2]),  clashOccurs=1, debug=debug) 
             
               if(res.resname)== 'PRO': continue

               res.add(Atom.Atom(name='H', coord=hCoordBB, bfactor=0., occupancy=1., altloc=' ', fullname='H', serial_number=lastSerial+i,element='H'))
               bbHcoordsInfo.append([lastSerial+i, i, res, hCoordBB])
               
               if debug ==1: 
                    fd.write(f"For serial number: {lastSerial+i}, i: {i}, res:{res}, and backbone hydrogen coords: {hCoordBB}\n")
                    fd.flush()

    if(debug ==1):
        stp.end_debug_file(__name__,sys._getframe().f_code.co_name, fd)         

    return lastSerial+i, bbHcoordsInfo


def place_hydrogens_ARG(res, lastSerial, debug=0):

    ''' objective: Placing side chain hydrogen for Arginine(ARG)
        input:-res: Arginine residue considered (who's side chain hydrogen(s) needs to be added)
              -lastSerial: serial number of the last atom in the structure
        output:-lastSerial+i:the updated serial number for the last atom in the structure
               -hydCoords: coordinates of the hydrogen atom(s) added
    ''' 

    hydCoords =[]
    CDCoord = res['CD'].coord
    NECoord = res['NE'].coord
    CZCoord = res['CZ'].coord
    NH1Coord = res['NH1'].coord
    NH2Coord = res['NH2'].coord

    hydCoords.append(compute_hydrogen_coords_sp2( np.array([ CDCoord, NECoord, CZCoord]), clashOccurs=1, debug=debug))
    hydCoords.append(compute_hydrogen_coords_sp2( np.array([NH2Coord, NH1Coord, CZCoord]), clashOccurs=0, debug=debug))
    hydCoords.append(compute_hydrogen_coords_sp2( np.array([NH1Coord, NH2Coord, CZCoord]), clashOccurs=0, debug=debug))

    lastSerial = lastSerial+1
    
    names = ['HE', 'HH11', 'HH12', 'HH21', 'HH22']
    hCoords = [hydCoords[0], hydCoords[1][0], hydCoords[1][1], hydCoords[2][0], hydCoords[2][1]]
    
    for i in range(len(names)):
       res.add(Bio.PDB.Atom.Atom(name=names[i], coord=hCoords[i], bfactor=0., occupancy=1., altloc=' ', fullname=names[i], serial_number=lastSerial+i,element='H'))

    if(debug ==1):
        angle1 = calc_angle(res['CD'].get_vector(), res['NE'].get_vector(), res['HE'].get_vector())*180/3.14
        angle2 = calc_angle(res['CZ'].get_vector(), res['NE'].get_vector(), res['HE'].get_vector())*180/3.14

        angle3 = calc_angle(res['CZ'].get_vector(), res['NH1'].get_vector(), res['HH11'].get_vector())*180/3.14
        angle4 = calc_angle(res['CZ'].get_vector(), res['NH1'].get_vector(), res['HH12'].get_vector())*180/3.14
        angle5 = calc_angle(res['HH11'].get_vector(), res['NH1'].get_vector(), res['HH12'].get_vector())*180/3.14

        angle6 = calc_angle(res['CZ'].get_vector(), res['NH2'].get_vector(), res['HH21'].get_vector())*180/3.14
        angle7 = calc_angle(res['CZ'].get_vector(), res['NH2'].get_vector(), res['HH22'].get_vector())*180/3.14
        angle8 = calc_angle(res['HH21'].get_vector(), res['NH2'].get_vector(), res['HH22'].get_vector())*180/3.14


        stp.append_to_debug( __name__, sys._getframe().f_code.co_name, f"Side chain hydrogens placed for residue: {res}, belonging to chain:{res.parent} .\n. Tha angle values in degrees are:\
                angle1: CD-NE-HE :{angle1}\n \
                angle2: CZ-NE-HE :{angle2}\n \
                angle3: CZ-NH1-HH11 :{angle3}\n \
                angle4: CZ-NH1-HH12 :{angle4}\n \
                angle5: HH11-NH1-HH12 :{angle5}\n \
                angle6: CZ-NH2-HH21 :{angle6}\n \
                angle7: CZ-NH2-HH22 :{angle7}\n \
                angle8: HH21-NH2-HH22 :{angle8}\n \
                ", debug=0 )

    return lastSerial+i, hydCoords




def place_hydrogens_ASN(res, lastSerial, debug=0):

    ''' 
        objective: Placing side chain hydrogen for Aspargine(ASN)
        input:-res: Aspargine residue considered (who's side chain hydrogen(s) needs to be added)
              -lastSerial: serial number of the last atom in the structure
        output:-lastSerial+i:the updated serial number for the last atom in the structure
               -hd2Coords: coordinates of the hydrogen atom(s) added
    ''' 

    hvys = np.array([res['OD1'].coord, res['ND2'].coord, res['CG'].coord  ])
    hd2Coords = compute_hydrogen_coords_sp2( hvys, clashOccurs=0, debug=debug)

    
    names = ['HD21','HD22']
    lastSerial = lastSerial+1

    for i in range(len(names)):
        res.add(Bio.PDB.Atom.Atom(name=names[i], coord=hd2Coords[i], bfactor=0., occupancy=1., altloc=' ', fullname=names[i], serial_number=lastSerial+i,element='H'))

    if(debug==1):
        stp.append_to_debug( __name__, sys._getframe().f_code.co_name, f"Side chain hydrogen placed, for {res}, added {len(names)} Hs, with s.no now at: {lastSerial+i} !", debug=0)
  
    return lastSerial+i, hd2Coords



def place_hydrogens_GLN(res, lastSerial, debug = 0):
    
    ''' objective: Placing side chain hydrogen for Glutamine(GLN)
        input:-res: the Glutamine residue considered (who's side chain hydrogen(s) needs to be added)
              -lastSerial: serial number of the last atom in the structure
        output:-lastSerial+i:the updated serial number for the last atom in the structure
               -he2Coords: coordinates of the hydrogen atom(s) added
    ''' 

    hvys = np.array([res['OE1'].coord, res['NE2'].coord, res['CD'].coord  ])

    he2Coords = compute_hydrogen_coords_sp2(hvys, clashOccurs=0, debug=debug)
    
    names = ['HE21','HE22']
    lastSerial = lastSerial+1

    for i in range(len(names)):
        res.add(Bio.PDB.Atom.Atom(name=names[i], coord=he2Coords[i], bfactor=0., occupancy=1., altloc=' ', fullname=names[i], serial_number=lastSerial+i,element='H'))
    
    if(debug==1):
        stp.append_to_debug( __name__, sys._getframe().f_code.co_name, f"Side chain hydrogen placed, for {res}, added {len(names)} Hs, with s.no now at: {lastSerial+i} !", debug=0)
  

    return lastSerial+i, he2Coords


def get_default_HH_lonepair_coord(hhCoords,reason,debug =0):
    '''
        objective: To provide a default value of hydrogen (HH) coordinates for Tyrosine for appropriate conditions(such as no surrounding close atoms)
        input:-hhCoords: the two possible hydrogen coordinates
              -reason: string that describes why one is using default hydrogen coordinate
        output:-hhCoord:the hydrogen coordinate picked
               -LPCoord:lone pair coordinates attached to OH in Tyrosine (picking the coordinate which is not picked as hydrogen coordinate)
    '''
    
    hhCoord = hhCoords[0]
    LPCoord = hhCoords[1]

    if(debug==1):
        stp.append_to_debug( __name__, sys._getframe().f_code.co_name, f"Going to default as: {reason} to eliminate the two solutions of TYR. Taking the first coordinate computed: {hhCoord} and second coordinate as LP coord: {LPCoord}", debug=0)
    return hhCoord, LPCoord


def set_HH_and_lonepair_coords_TYR(aboveSp2, sp2, hhCoords, debug=0):
    '''
    objective: -To set side chain hydrogen(HH) and lone pair attached to oxygen(OH) on Tyrosine
               -Need to figure out which coordinate is assigned to hydrogen and which one for lone pair
    input:
          -aboveSp2: The atom(non hydrogen) attached to the sp2 atom
          -sp2: The sp2 atom attached to the hydrogen
          -hhCoords: Coordinates of hydrogens/LP. 
    output:-hhCoord: Coordinate for hydrogen(HH)
           -LPCoord: Coordinate for the lone pair on OH oxygen
    '''
    
    if(debug==1):
        stp.start_debug_file( __name__, sys._getframe().f_code.co_name)
        fDebugName = stp.get_debug_file_name(debug=0)
        fd = open(fDebugName, "a")

    res = sp2.parent
    structure = res.parent.parent.parent
    
    #get all donor acceptor list
    customListAll = stp.get_donor_acceptor_list(structure, aaType = 'DONOR_ACCEPTOR_BOTH_TBD', debug=debug)

    #get only donor acceptor list that are known wrt to self atom(backbones are known and self atom behaviors are known)
    customList = stp.get_known_donor_acceptor_list_for_one_atom(structure, sp2, aaType = 'DONOR_ACCEPTOR', debug=debug)
    allCloseAtoms_DoAcBoT = cats.get_all_close_atom_info_for_one_atom(sp2, customListAll, debug=debug)
    
    #If there are no known donor/acceptor/both or TBD atom-it can pick up default value
    if(not allCloseAtoms_DoAcBoT):
        hhCoord, LPCoord = get_default_HH_lonepair_coord(hhCoords, reason=f'No Close Atoms of any behaviors found for {res}, {res.id}', debug=debug)
        return hhCoord,LPCoord

    #If there are no known donor/acceptor then do not assign coordinate values for hydrogen and lone pairs!
    if(not customList):

        if(debug==1):
            fd.write(f"\n NO KNOWN CLOSE ATOMS FOUND for {res}, {res.id}\n")
            fd.flush()

        hCoord = []
        LPCoord = []
        return hCoord, LPCoord

    ##Get a list of all close Atoms (which are donor, acceptor, and both) for the given hvy atom
    allCloseAtoms = cats.get_all_close_atom_info_for_one_atom(sp2, customList, debug=debug) 
    ##if there are absolutely no close atoms-then pick up default values
    if(np.shape(allCloseAtoms)[0]<2):
        hhCoord, LPCoord = get_default_HH_lonepair_coord(hhCoords, reason=f'No Close Atoms found for {res}, {res.id}', debug=debug)
        return hhCoord,LPCoord
     
    mySp2 = mra.my_atom(sp2)
    sp2_vec = sp2.get_vector() ##atom2
    aboveSp2_vec = aboveSp2.get_vector()##atom4
    enVal =[]
    enSumList = [] 

    #Loop over the hydrogens coords-i.e going to be 2-(hhcoord/lpcoord) and find minimum energy interaction values to assign it position
    for j in range(np.shape(hhCoords)[0]):
        hh_coord= hhCoords[j]
        hh_vec = Vector(hh_coord)##atom3
        lp_coord = hhCoords[abs(j-1)] ##BEhavior AS ACC. LP VEC!!
        lp_vec = Vector(hhCoords[abs(j-1)]) ##BEhavior AS ACC. LP VEC!!

        enSum = 0
        if(debug ==1): 
            fd.write(f"close Atom now: {allCloseAtoms}\n")
            fd.flush()
        #Loop over all the close atoms to find energy interaction with OH-HH/OH-LP
        for i in range(1, np.shape(allCloseAtoms)[0]):

            closeAt = allCloseAtoms[i][0]
            myCloseAt = mra.my_atom(allCloseAtoms[i][0])
            enValDon = 0
            enValAcc = 0

            if(myCloseAt.get_behavior().abbrev == 'ac'):
                if(debug==1):fd.close()
                ##Considering donor atom associated with reference atom
                enValDon, enSumDon = cats.compute_energy_as_donor(closeAt, hhCoords[j], sp2, attractive = 1, atype='SP2',  chV_levelVal = 'level_00_chV_00_structureNum_00', debug=debug )
                if(debug==1):fd = open(fDebugName, "a")
                if(enValDon == []):
                    if(debug==1):
                        fd.write(f"\n I am DONOR and I will continue as no energyVal found\n")
                        fd.flush()
                    continue 
                enSum = enSum + enSumDon
                if(debug==1):fd.close()
                #Considering acceptor atom associated with reference atom
                enValAcc, enSumAcc = cats.compute_energy_as_acceptor(sp2, lp_vec, closeAt,attractive = 0, atype = 'SP2',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)
                if(debug==1):fd = open(fDebugName, "a")
                enSum = enSum + enSumAcc

            if(myCloseAt.get_behavior().abbrev == 'do'):
                ##Considering acceptor atom associated with reference atom
                if(debug==1):fd.close()
                enValAcc, enSumAcc = cats.compute_energy_as_acceptor(sp2, lp_vec, closeAt, attractive = 1, atype='SP2',  chV_levelVal ='level_00_chV_00_structureNum_00', debug=debug)
                if(debug==1):fd = open(fDebugName, "a")
                if not enValAcc:
                    if(debug==1):
                        fd.write(f"\n I am ACC and will continue as no energyVal found \n")
                    continue
                enSum = enSum + enSumAcc
                ##Considering donor atom associated with reference atom
                if(debug==1):fd.close()
                enValDon, enSumDon = cats.compute_energy_as_donor(closeAt, hhCoords[j], sp2, attractive =0, atype = 'SP2',  chV_levelVal ='level_00_chV_00_structureNum_00', debug =debug)
                if(debug==1):fd = open(fDebugName, "a")
                enSum = enSum + enSumDon
        enSumList.append([hhCoords[j], enSum])
        
    #Array of hydrogen coordinates and energy sum
    enValArr = np.array(enSumList, dtype=object)
    #If no ener
    if not enValArr[:,-1].all():
        hhCoord, LPCoord = get_default_HH_lonepair_coord(hhCoords, reason='All energy value = 0! Asumming no interaction-picking default', debug=debug )
        return hhCoord,LPCoord

#   ##MinInd for the energy val index
    minInd = np.argmin(enValArr[:,-1])
    ##pick up the hydrogen coordinate corresponding to that
    hhCoord = enValArr[minInd, -2]

    cats.check_energy_range(enValArr[minInd, -1], stateSet = False, debug=debug)

    #Fix the lone pair coordinate(by elimination of the hydrogen coordinate)
    if(all(abs(hhCoords[0]-hhCoord))<0.0001 ):
        LPCoord = hhCoords[1]
    else:
        LPCoord = hhCoords[0]

    if(debug==1):
        stp.end_debug_file(__name__,sys._getframe().f_code.co_name, fd) 

    return hhCoord,LPCoord  
#
def place_hydrogens_TYR(res, lastSerial, debug=0):
           
    ''' objective: Placing side chain hydrogen for Tyrosine(TYR)
        input:-res: the Tyrosine residue considered (who's side chain hydrogen(s) needs to be added)
              -lastSerial: serial number of the last atom in the structure
        output:-lastSerial+i:the updated serial number for the last atom in the structure
               -hLPCoords: coordinates of the hydrogen atom and lone pair atom added

    ''' 
    
    hvys = np.array([res['CE1'].coord, res['OH'].coord, res['CZ'].coord  ])
    hhCoords = compute_hydrogen_coords_sp2(hvys, clashOccurs=0, debug=debug)
    ##eliminate on of the hydrogens based off the energy calc
    hhCoord, LPCoord = set_HH_and_lonepair_coords_TYR(res['CZ'], res['OH'], hhCoords, debug=debug) 
    
    if(not any(hhCoord)):
        if(debug==1):
            stp.append_to_debug( __name__, sys._getframe().f_code.co_name, f"NO CLOSE ATOMS FOUND for {res}, {res.id}", debug=0)
        hCoord = []
        return lastSerial, hCoord

    else: 
        res.add(Bio.PDB.Atom.Atom(name='HH', coord=hhCoord, bfactor=0., occupancy=1., altloc=' ', fullname='HH', serial_number=lastSerial+1,element='H'))
        res.add(Bio.PDB.Atom.Atom(name='LP3', coord=LPCoord, bfactor=0., occupancy=1., altloc=' ', fullname='LP3', serial_number=lastSerial+2,element='LP'))

        hhLP= [hhCoord, LPCoord]
        hLPCoords = np.reshape(hhLP,(2,3))

        if(debug==1):
            stp.append_to_debug_less_detail(f"Side chain hydrogen placed, for {res}, added HH hydrogen and LP3 lone pair, with s.no now at: {lastSerial+1} !", debug=0)
        ##To make it more like the SP3-SER/THR otherwise keep hhCoord because thats how we are treating it in the main loop
        return lastSerial+2, hLPCoords


def place_hydrogens_TRP(res, lastSerial, debug=0):
    ''' objective: Placing side chain hydrogen for Tryptophan (TYR)
        
        input:-res: the Tryptophan residue considered (who's side chain hydrogen(s) needs to be added)
              -lastSerial: serial number of the last atom in the structure
        output:-lastSerial+i:the updated serial number for the last atom in the structure
               -he1Coords: coordinates of the hydrogen atom(s) added
        ''' 
    
    hvys = np.array([res['CD1'].coord, res['NE1'].coord, res['CE2'].coord  ])

    he1Coord = compute_hydrogen_coords_sp2(hvys, clashOccurs=1, debug=debug)
    res.add(Bio.PDB.Atom.Atom(name='HE1', coord=he1Coord, bfactor=0., occupancy=1., altloc=' ', fullname='HE1', serial_number=lastSerial+1,element='H'))

    if(debug==1):
        stp.append_to_debug( __name__, sys._getframe().f_code.co_name, f"Side chain hydrogen placed, for {res}, added HE1 hydrogen and LP3 lone pair, with s.no now at: {lastSerial+1} !", debug=0)

    return lastSerial+1, he1Coord


def place_hydrogens_HID(res, lastSerial, debug = 0):
            
    '''objective: Placing side chain hydrogen for Histidine (HID)
       input:-res: the HID residue considered (who's side chain hydrogen(s) needs to be added)
              -lastSerial: serial number of the last atom in the structure
        output:-lastSerial+i:the updated serial number for the last atom in the structure
               -hd1Coords: coordinates of the hydrogen atom(s) added
    ''' 
 


    hvys = np.array([res['CG'].coord, res['ND1'].coord, res['CE1'].coord  ])

    hd1Coord = compute_hydrogen_coords_sp2(hvys, clashOccurs=1, debug=debug)
    res.add(Bio.PDB.Atom.Atom(name='HD1', coord=hd1Coord, bfactor=0., occupancy=1., altloc=' ', fullname='HD1', serial_number=lastSerial+1,element='H'))
    
    if(debug==1):
        stp.append_to_debug( __name__, sys._getframe().f_code.co_name, f"Side chain hydrogen placed, for {res}, added HD1 hydrogen, with s.no now at: {lastSerial+1} !", debug=0)

    return lastSerial+1, hd1Coord


def place_hydrogens_HIE(res, lastSerial, debug=0):
         
    '''objective: Placing side chain hydrogen for Histidine (HIE)
       input:-res: the HIE residue considered (who's side chain hydrogen(s) needs to be added)
              -lastSerial: serial number of the last atom in the structure

       output:-lastSerial+i:the updated serial number for the last atom in the structure
               -he2Coords: coordinates of the hydrogen atom(s) added
    ''' 
    hvys = np.array([res['CD2'].coord, res['NE2'].coord, res['CE1'].coord  ])
    he2Coord = compute_hydrogen_coords_sp2(hvys, clashOccurs=1, debug=debug)

    res.add(Bio.PDB.Atom.Atom(name='HE2', coord=he2Coord, bfactor=0., occupancy=1., altloc=' ', fullname='HE2', serial_number=lastSerial+1,element='H'))

    if(debug==1):
        stp.append_to_debug( __name__, sys._getframe().f_code.co_name, f"Side chain hydrogen placed, for {res}, added HE2 hydrogen, with s.no now at: {lastSerial+1} !", debug=0)

   
    return lastSerial+1, he2Coord

def place_hydrogens_HIP(res, lastSerial, debug = 0):

    ''' objective: Placing side chain hydrogen for Histidine (HIP)
        HIP has features of both HID and HIE 
        input:-res: the HIP residue considered (who's side chain hydrogen(s) needs to be added)
              -lastSerial: serial number of the last atom in the structure

        output:-lastSerial+i:the updated serial number for the last atom in the structure
               -hipCoords: coordinates of the hydrogen atom(s) added

        '''  
    lastSerial, hCoord_HID= place_hydrogens_HID(res, lastSerial, debug=debug) 
    lastSerial, hCoord_HIE= place_hydrogens_HIE(res, lastSerial, debug=debug)

    hipCoords = np.array([hCoord_HID, hCoord_HIE]) 

##The return value will not increase for last serial as each placeHydrogens_ functions will increase it correctly
    return lastSerial, hipCoords


def add_sp2_sidechain_hydrogens(structure,lastSerial, debug =0):
    ''' Objective: To add all SP2 Hydrogen SideChains to a given structure
        input:-structure:the structure in consideration (of which all residues Sidechain needs hydrogen)
           -modelID: the modelID in consideration (of which all residues side chain needs hydrogen)
           -chainID: the chainID in consideration (of which all residues side chain need hydrogen)
           -lastSerial: the serial number of the last atom in the structure

        Output: lastSerial: updated serial number for the atom added
                hCoord: hydrogen coordinate of the last residue computed

        '''
    
    res_dict = {'ARG': place_hydrogens_ARG, 'ASN':place_hydrogens_ASN, 'GLN': place_hydrogens_GLN, 'TYR': place_hydrogens_TYR, 'TRP': place_hydrogens_TRP, 'HID': place_hydrogens_HID, 'HIE': place_hydrogens_HIE, 'HIP': place_hydrogens_HIP}


    for res in structure.get_residues():
        if(res.resname =='TYR'): continue 
        try:
            lastSerial, hCoord = res_dict[res.resname](res, lastSerial, debug=debug)
        except KeyError:
            if(debug==1):
                stp.append_to_debug_less_detail(f"No Side chain Hydrogens for: {res.resname} with {res.id[1]} and chain: {res.parent}. Adding Side chain hydrogen only for: ARG, ASN, GLN, TYR, TRP, HID, HIE, HIP.\n", debug=0 )
            pass
    return lastSerial, hCoord 
             

def add_sp2_sidechain_lonepairs(structure, lastSerial, debug =0):

    ''' Objective: To add all SP2 LP SideChains to a given structure
        input:-structure:the structure in consideration (of which all residues Sidechain needs LP)
           -modelID: the modelID in consideration (of which all residues side chain needs LP)
           -chainID: the chainID in consideration (of which all residues side chain need LP)
           -lastSerial: the serial number of the last atom in the structure

        Output: lastSerial: updated serial number for the atom added
                lpCoord: lone pair atom coordinate of the last residue computed

        '''

    LP_dict = {'ASP': [mc.hvysForLPsASP, mc.LPSCnamesASP], 'GLU': [mc.hvysForLPsGLU, mc.LPSCnamesGLU],
                'ASN': [mc.hvysForLPsASN, mc.LPSCnamesASN], 'GLN': [mc.hvysForLPsGLN, mc.LPSCnamesGLN],
                'HIE': [mc.hvysForLPsHIE, mc.LPSCnamesHIE], 'HID': [mc.hvysForLPsHID, mc.LPSCnamesHID]
    }


    for res in structure.get_residues():
        try:
            hvys = LP_dict[res.resname][0]
            LPnameAll = LP_dict[res.resname][1]
            lastSerial, lpCoord = place_lonepair(res, lastSerial, hvys, LPnameAll, debug=debug)
        except KeyError:
            if(debug==1):
                stp.append_to_debug_less_detail(f"No Lone pairs for: {res.resname} with {res.id[1]} and chain: {res.parent}. Adding side chain lone pairs only for: ASP, GLU, ASN, GLN, HIE, HID.\n", debug=0 )
            pass

    return lastSerial, lpCoord

