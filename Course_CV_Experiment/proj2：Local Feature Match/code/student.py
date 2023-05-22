import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops


def get_interest_points(image, feature_width):
    '''
    Returns a set of interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # Convert the image to grayscale if it is color
    if image.ndim == 3:
        image = img_as_int(image)

    # Compute the Harris response map
    harris_response = feature.corner_harris(image)

    # Find local maxima in the response map as interest points
    coordinates = feature.peak_local_max(harris_response, min_distance=feature_width//2,
                                         exclude_border=True)

    # Filter out interest points with low response values and those near the image boundary
    mask = (coordinates[:, 0] >= feature_width//2) & (coordinates[:, 0] < image.shape[0] - feature_width//2) \
           & (coordinates[:, 1] >= feature_width//2) & (coordinates[:, 1] < image.shape[1] - feature_width//2) \
           & (harris_response[coordinates[:, 0], coordinates[:, 1]] > 0.01)
    xs = coordinates[mask, 1]
    ys = coordinates[mask, 0]

    return xs, ys
    

def get_features(image, x, y, feature_width):
    '''
    Returns a set of feature descriptors for a given set of interest points.

    (Please note that we reccomend implementing this function after you have implemented
    match_features)

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each descriptor_window_image_width/4.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # Calculate the number of cells in the feature grid
    num_cells = feature_width // 4
    
    # Calculate the size of each cell
    cell_size = feature_width // num_cells
    
    # Initialize an empty array to store the computed features
    features = np.zeros((len(x), num_cells * num_cells * 8))
    
    for i in range(len(x)):
        # Extract the patch around the interest point
        patch = image[y[i]-feature_width//2:y[i]+feature_width//2,
                      x[i]-feature_width//2:x[i]+feature_width//2]
        
        # Normalize the patch
        patch = (patch - np.mean(patch)) / np.std(patch)

        # Apply a threshold to remove low values
        patch[patch < 0.2*np.max(patch)] = 0

        # Normalize the patch again
        patch = (patch - np.mean(patch)) / np.std(patch)
        
        # Compute gradients in x and y directions
        gradients_y = filters.sobel_v(patch)
        gradients_x = filters.sobel_h(patch)
        
        # Compute gradient magnitudes and orientations
        magnitudes = np.sqrt(gradients_x**2 + gradients_y**2)
        orientations = np.arctan2(gradients_y, gradients_x)
        orientations = np.degrees(orientations) % 360
        
        # Split the patch into cells and compute histograms
        for row in range(num_cells):
            for col in range(num_cells):
                cell = magnitudes[row*cell_size:(row+1)*cell_size,
                                  col*cell_size:(col+1)*cell_size]
                cell_orientations = orientations[row*cell_size:(row+1)*cell_size,
                                                 col*cell_size:(col+1)*cell_size]
                histogram, _ = np.histogram(cell_orientations, bins=8,
                                            range=(0, 360), weights=cell)
                
                # Append the histogram to the feature vector
                features[i, (row*num_cells+col)*8:(row*num_cells+col+1)*8] = histogram
        
        # Check if the feature vector is not a zero vector
        if np.linalg.norm(features[i]) != 0:
            # Normalize the feature vector
            features[i] /= np.linalg.norm(features[i])
    
    return features

def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - zip (python built in function)

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # Initialize empty lists to store matches and confidences
    matches = []
    confidences = []
    
    # Iterate over features in im1_features
    for i in range(len(im1_features)):
        feature1 = im1_features[i]
        
        # Compute Euclidean distances between feature1 and all features in im2_features
        distances = np.linalg.norm(im2_features - feature1, axis=1)
        
        # Sort the distances and find the two closest matches
        sorted_indices = np.argsort(distances)
        closest_index = sorted_indices[0]
        second_closest_index = sorted_indices[1]
        
        # Apply the Nearest Neighbor Distance Ratio (NNDR) Test
        nndr = distances[closest_index] / distances[second_closest_index]
        
        if nndr < 0.8:  # NNDR threshold (can be adjusted)
            matches.append([i, closest_index])
            confidences.append(1.0 - nndr)
    
    # Convert matches and confidences to NumPy arrays
    matches = np.asarray(matches)
    confidences = np.asarray(confidences)
    
    return matches, confidences
