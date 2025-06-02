import numpy as np

def mod(x, debug=0): 
    
    ''' objective: modulus of a vector is computed. Use this for computing distances
        input:x: numpy array/vector
        output: mod value
    '''
    
    mod_val = np.sqrt(x.dot(x))
    return mod_val


def unitVec(x, debug=0):

    '''objective: Finding a unit vector by dividing with the modulus of the vector 
       input:x-a vector you want to normalize
       output: unit vector
    '''

    return  x/mod(x)

def rotMatrix(theta, debug=0):

    '''objecitve: rotation about Z axis. Provide theta input in degrees
       input: theta is the degree about which we rotate about Z axis. (Provide input in degrees)
       output:rotation matrix
    '''
    thetaRad = np.radians(theta)
    cosTheta = np.cos(thetaRad)
    sinTheta = np.sin(thetaRad)

    rotMat = np.array([[cosTheta, sinTheta, 0], [-sinTheta, cosTheta, 0],[0,0,1]])

    return rotMat

def rotMatrixAboutY(theta, debug=0):

    '''objecitve: rotation about Y axis. Provide theta input in degrees
       input: theta is the degree about which we rotate about Z axis (Provide input in degrees)
       output:rotation matrix
    '''


    thetaRad = np.radians(theta)
    cosTheta = np.cos(thetaRad)
    sinTheta = np.sin(thetaRad)
    

    rotMatY = np.array([[cosTheta, 0, sinTheta ], [0,1,0], [-sinTheta, 0, cosTheta] ])


    return rotMatY


def translateCoord( B, A, debug=0):

    ''' objective: Translating coords of points from A to B 
        input:-B:coords of point B (by which you want to shift all points A)
              -A: coords of point A
        output: new coords
    '''

    return A-B



