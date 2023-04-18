import cv2
import numpy as np
import dlib

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(detector, predictor, img):
    
    '''
    This function first use `detector` to localize face bbox and then use `predictor` to detect landmarks (68 points, dtype: np.array).
    
    Inputs: 
        detector: a dlib face detector
        predictor: a dlib landmark detector, require the input as face detected by detector
        img: input image
        
    Outputs:
        landmarks: 68 detected landmark points, dtype: np.array

    '''
    
    #TODO: Implement this function!
    # Your Code to detect faces
    faces = None
    
    if len(faces) > 1:
        raise TooManyFaces
    if len(faces) == 0:
        raise NoFaces
    
    # Your Code to detect landmarks
    landmarks = None

    return landmarks

def get_face_mask(img, landmarks):
    
    '''
    This function gets the face mask according to landmarks.
    
    Inputs: 
        img: input image
        landmarks: 68 detected landmark points, dtype: np.array
        
    Outputs:
        convexhull: face convexhull
        mask: face mask 

    '''
    
    #TODO: Implement this function!
    convexhull, mask = None, None

    return convexhull, mask

def get_delaunay_triangulation(landmarks, convexhull):
    
    '''
    This function gets the face mesh triangulation according to landmarks.
    
    Inputs: 
        landmarks: 68 detected landmark points, dtype: np.array
        convexhull: face convexhull
        
    Outputs:
        triangles: face triangles 
    '''
    
    #TODO: Implement this function!
    triangles = None

    return triangles

def transformation_from_landmarks(target_landmarks, source_landmarks):
    '''
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    
    Inputs: 
        target_landmarks: 68 detected landmark points of the target face, dtype: np.array
        source_landmarks: 68 detected landmark points of the source face that need to be warped, dtype: np.array
        
    Outputs:
        triangles: face triangles 
    '''
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    
    #TODO: Implement this function!
    M = None
    
    return M

def warp_img(img, M, target_shape):
    '''
    This function utilizes the affine transformation matrix M to transform the img.
    
    Inputs: 
        img: input image (np.array) need to be warped.
        M: affine transformation matrix.
        target_shape: the image shape of target image
        
    Outputs:
        warped_img: warped image.
    
    '''
    
    #TODO: Implement this function!
    warped_img = None
    
    return warped_img