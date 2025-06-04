"""
This module contains all the mathematical computations used in this code.

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

def mod(x):
    
    """
        objective: modulus of a vector is computed
        input:x: numpy array/vector
        output: mod value
    """
    
    mod_val = np.sqrt(x.dot(x))
    return mod_val


def get_unit_vector(x):

    """
       objective: Finding a unit vector by dividing with the modulus of the vector 
       input:x-a vector you want to normalize
       output: unit vector
    """

    return  x/mod(x)

def get_rotation_matrix(theta):

    """
       objective: rotation about Z axis. Provide theta input in degrees
       input: theta is the degree about which we rotate about Z axis. (Provide input in degrees)
       output:rotation matrix
    """
    thetaRad = np.radians(theta)
    cosTheta = np.cos(thetaRad)
    sinTheta = np.sin(thetaRad)

    rotMat = np.array([[cosTheta, sinTheta, 0], [-sinTheta, cosTheta, 0],[0,0,1]])

    return rotMat

def get_rotation_matrix_about_Yaxis(theta):

    """
        objective: rotation about Y axis. Provide theta input in degrees
       input: theta is the degree about which we rotate about Z axis (Provide input in degrees)
       output:rotation matrix
    """


    thetaRad = np.radians(theta)
    cosTheta = np.cos(thetaRad)
    sinTheta = np.sin(thetaRad)
    

    rotMatY = np.array([[cosTheta, 0, sinTheta ], [0,1,0], [-sinTheta, 0, cosTheta] ])


    return rotMatY


def translate_coord( B, A):

    """ 
        objective: Translating coordinates of points from A to B
        input:-B:coordinates of point B (by which you want to shift all points A)
              -A: coordinates of point A
        output: new coordinates
    """

    return A-B



