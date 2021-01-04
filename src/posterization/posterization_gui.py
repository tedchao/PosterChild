import sys
import numpy as np
import PIL.Image
import PIL.ImageTk

import gco

from skimage import color
from sklearn.cluster import KMeans
from scipy.optimize import *
import cv2

from numba import jit
from math import *

import random
import time

from . import simplepalettes
from .simplify_convexhull import *


def get_candidate_colors_and_neighbor_list( mesh, weight_list, num_colors, num_blend ):
    '''
    Given: palette (np array of colors), list of weights for each color (2D list),
    number of colors, number of blends.
    
    Return: A neighboring information for different blends and palette. (2D list),
    list of weights for each color (2D list), a list of candidate colors.
    '''
    size = 1 / ( num_blend + 1 )
    a, b, c = 0.5, 0.25, 0.75
    blends = [[a, 1-a], [b, 1-b], [c, 1-c]]
    
    # store the neighbors
    # 2D list
    neighbor_list = []
    for i in range( num_colors ):
        neighbor_list.append( [] )


    # discrete blendings
    candidate_colors = mesh
    pos_iter = num_colors
    
    for i in range( num_colors ):
        for j in range( i+1, num_colors ):
            
            # add blended colors from i-position to j-position
            for k in range( num_blend ):
                candidate_colors = np.vstack( ( candidate_colors, ( 1 - ( k + 1 ) * size ) * candidate_colors[i] + ( k + 1 ) *  size * candidate_colors[j] ) ) 
                
                if k == 0 and k == num_blend - 1:
                    neighbor_list.append( [i, j] )  # only 1 blend
                    neighbor_list[i].append( pos_iter ) 
                    neighbor_list[j].append( pos_iter )
                    
                elif k == 0 or k == num_blend - 1:
                    if k == 0:
                        neighbor_list.append( [i, pos_iter + 1] )   # first head
                        neighbor_list[i].append( pos_iter ) 
                    else:
                        neighbor_list.append( [j, pos_iter + num_blend - 2] )   # last tail
                        neighbor_list[j].append( pos_iter + num_blend - 1 ) 
                else:
                    #if k != 0 and k != num_blend - 1:
                    neighbor_list.append( [pos_iter + k - 1, pos_iter + k + 1] )

            # add palette weight for each color to weight list
            for s in range( num_blend ):
                weights = num_colors * [0]
                weights[i] = 1 - ( s + 1 ) * size
                weights[j] = ( s + 1 ) *  size
                weight_list.append( weights )
                
            pos_iter += num_blend
            
    return candidate_colors, neighbor_list, np.array( weight_list )

    
@jit(nopython=True)
def get_unary( image_arr, candidate_colors ):
    '''
    Given: original image np array, list of candidate colors.
    
    Return: Unary energy for MLO.
    '''
    
    # unary.shape = (486, 864, len(labels)) ideally!
    unary = np.zeros( ( image_arr.shape[0], image_arr.shape[1], len( candidate_colors ) ) )
    
    for i in range( len( candidate_colors ) ):
        dist_square = ( candidate_colors[i] - image_arr ) ** 2
        unary[:, :, i] = np.sqrt( np.sum( dist_square, axis = 2 ) )
    return unary


def get_binary( neighbor_list, weight_list, candidate_colors, option = 1, penalization = 0.7 ):
    '''
    Given: list of neighboring information (2D list), list of weights for each color (2D list),
    list of candidate colors, regularization term, and option. (option=1: Simple Pott's potential,
    option=2: differences in RGB space distance between candidate colors.)
    
    Return: Binary energy for MLO.
    '''
    
    # 1 if neighbor, (inf) if not neighbor, diag(binary) = 0
    if option == 1:
        binary = np.ones( ( len( neighbor_list ), len( neighbor_list ) ) ) * np.inf
        
        for i in range( len( candidate_colors ) ):
            for j in range( len( candidate_colors ) ):
                if i == j:
                    binary[i][j] = 0
                    
        for i in range( len( neighbor_list ) ):
            for j in range( len( neighbor_list[i] ) ):
                binary[i][neighbor_list[i][j]] = 0.5
                
    # half of the L1 norm distance based on weights on each palette
    if option == 2:
        binary = np.zeros( ( len( neighbor_list ), len( neighbor_list ) ) )
        for i in range( len( candidate_colors ) ):
            for j in range( len( candidate_colors ) ):
                if i != j:
                    binary[i][j] = penalization * np.linalg.norm( ( candidate_colors[i] - candidate_colors[j] ), ord=2 )
                    
    return binary

@jit(nopython=True)
def get_nonrepeated_labels( labels ):
    '''
    Given: list of labels (width x height, 1).
        
    Return: list of nonrepeated used labels.
    '''
    
    visited = []
    for label in labels:
        if label not in visited:
            visited.append( label )
    return visited


def optimize_weight( labels, nonrepeated_labels, weight_list, palette, image_arr ):
    '''
    Given: list of labels (width x height, 1), list of nonrepeated used labels, 
    list of weights for each color (2D list), np array of palette colors, original image.
    
    Return: list of "locally" optimized weights for each color (2D list).
    '''
    
    @jit(nopython=True)
    def frobenius_inner_product( matrix1, matrix2 ):
        '''
        Given: Two matrices.
        
        Return: Frobenius inner product of two matrices.
        '''
        assert matrix1.shape == matrix2.shape
        return np.trace( matrix1.T @ matrix2 )
    
    print("Local optimization on weights...")
    
    width = image_arr.shape[0]
    height = image_arr.shape[1]
    
    labels = np.asfarray( labels ).reshape( width, height, 1 )
    palette_labels = []
    for i in range( len( palette ) ):
        palette_labels.append( i )
        
    for label in nonrepeated_labels:
        if label not in palette_labels:
            F_label = []
            for i in range( width ):
                for j in range( height ):
                    if labels[i][j] == label:
                        F_label.append( image_arr[i, j, :] )
                        
            # quadratic programming            
            F_prime = np.array( F_label )
            pixel_size = F_prime.shape[0]           
            
            w_lst = weight_list[label]
            
            Pi_color = palette[np.nonzero(w_lst)[0][0]][np.newaxis]    # shape:1x3
            Pj_color = palette[np.nonzero(w_lst)[0][1]][np.newaxis]
            
            Is = np.ones(pixel_size)[np.newaxis]
            Pi = Is.T @ Pi_color
            Pj = Is.T @ Pj_color
            
            # some coefficients
            Pi_Pi = frobenius_inner_product( Pi, Pi )
            Pj_Pj = frobenius_inner_product( Pj, Pj )
            Pi_Pj = frobenius_inner_product( Pi, Pj )
            F_Pi = frobenius_inner_product( F_prime, Pi )
            F_Pj = frobenius_inner_product( F_prime, Pj )
            
            a = frobenius_inner_product( Pi - Pj, Pi - Pj )
            b = 2 * ( Pi_Pj - F_Pi + F_Pj - Pj_Pj )
            
            x = max( 0, min( 1, -b / ( 2 * a ) ) )
            
            nonzero_pos = np.nonzero( w_lst )[0]
            weight_list[label][nonzero_pos[0]] = x
            weight_list[label][nonzero_pos[1]] = 1-x
            
    return weight_list


def get_kmeans_cluster_image( num_clusters, img_arr, width, height, final_colors = None, weight_ls = None, option = 1 ):
    '''
    Given: predefined number of clusters, np array of images (shape: width x hegith, 3),
    image width, image height, list of final color, list of weights for each color
    and option. (option=1: clustering in RGB space, option=2: clustering in RGBXY space.)    
    
    Return: A reconstructed image after clustering.
    '''
    
    @jit(nopython=True)
    def get_num_clusters( weight_ls ):
        '''
        Given: list of weights for each selected colors from MLO.   
        
        Return: Number of colors that are not located at the same blended line.
        '''
        nonzero_indices = []
        for weights in weight_ls:
            w_nonzero = [i for i, e in enumerate( weights ) if e != 0]
            if w_nonzero not in nonzero_indices:
                nonzero_indices.append( w_nonzero )
                
        return len( nonzero_indices )
    
    @jit(nopython=True)
    def RGB_to_RGBXY( img, width, height ):
        '''
        Given: np array of images (shape: width x hegith, 3), image width, image height.   
        
        Return: An image with RGBXY. (shape: width x hegith, 5)
        '''
        
        img_rgbxy = np.zeros( ( img.shape[0], 5 ) )
        img = np.asfarray( img ).reshape( width, height, 3 )
        img_rgbxy = np.asfarray( img_rgbxy ).reshape( width, height, 5 )
        
        for i in range( width ):
            for j in range( height ):
                img_rgbxy[i, j, :3] = img[i, j]
                img_rgbxy[i, j, 3:5] = ( 1 / 600 ) * np.array( [i, j] )
        img_rgbxy = img_rgbxy.reshape( ( -1, 5 ) )
        
        return img_rgbxy
    
    # if we want to perform clustering in RGBXY space
    if option == 2:
        num_clusters = get_num_clusters( weight_ls )
        if num_clusters < len( final_colors ) - 10:
            num_clusters = len( final_colors ) - 10
        img_arr_rgbxy = RGB_to_RGBXY( img_arr, width, height )
        print( 'Final colors being used after Kmeans RGBXY:', num_clusters )
        
    if option == 2:
        kmeans = KMeans( n_clusters = num_clusters, random_state = 0 ).fit( img_arr_rgbxy )
    else:
        kmeans = KMeans( n_clusters = num_clusters, random_state = 0 ).fit( img_arr )
        
    kmeans_labels = kmeans.labels_
    clusters = kmeans.cluster_centers_
    
    for i in range( kmeans_labels.size ):
        if option == 2: # RGBXY space
            img_arr[i, :] = clusters[kmeans_labels[i]][:3]
        else: # RGB space
            img_arr[i, :] = clusters[kmeans_labels[i]]
    img_arr = np.asfarray( img_arr ).reshape( width, height, 3 )
    
    if option == 2:
        return img_arr, kmeans_labels, clusters
    else:
        return img_arr
    
    
def save_additive_mixing_layers( add_mix_layers, width, height, palette, save_path ):
    '''
    Given:
        add_mix_layers: additive mixing layers for each palette color (per-pixel weights list).
        width: image width.
        height: image height.
        palette: palette colors.
        save_path: saving path for additive mixing layers.
    
    Return: Nothing to return.
    '''
    
    def get_additive_mixing_layers( add_mix_layers, width, height, palette, color ):
        
        # select painting ink first
        paint = palette[color]
        
        # initialize layer
        img_add_mix = np.zeros( [width, height, 4] )
        add_mix_layers = add_mix_layers.reshape( width, height, len( palette ) )
        for i in range( width ):
            for j in range( height ):
                img_add_mix[i, j, :3] = paint
                img_add_mix[i, j, 3] = add_mix_layers[i, j, color]
                
        return img_add_mix
    
    for i in range( palette.shape[0] ):
        img_add_mix = get_additive_mixing_layers( add_mix_layers, width, height, palette, i )
        Image.fromarray( np.clip( 0, 255, img_add_mix * 255. ).astype( np.uint8 ), 'RGBA' ).save( save_path + '-' + str( i ) + '.png' )
        
        
def posterization( input_img_path, image_og, image_arr, num_colors, num_blend = 3, penalization = 0.7 ):
    '''
    Given:
        input_img_path: path for input image.
        image_og: original given image.
        image_arr: An n-rows by m-columns array of RGB colors (after Kmeans RGB).
        num_colors: number of palette colors for the poster.
        num_blend: number of blended colors chose from 2 palette.
    
    Return: A posterized image.
    '''
    
    assert len( image_og.shape ) == 3
    assert len( image_arr.shape ) == 3
    assert num_colors == int( num_colors ) and num_colors > 0
    
    width = image_og.shape[0]
    height = image_og.shape[1]
    
    def get_palette( path, image_arr, num_colors ):
        '''
        Given: path for input image, an n-rows by m-columns array of RGB colors (after Kmeans RGB).
        
        Return: Simplified convexhull for clustered input image (numpy array).
        '''
        
        # reshape image array into scipy.convexhull
        img_reshape = image_arr.reshape( ( -1, 3 ) )
        
        relative_mesh = img_reshape - img_reshape[0]    # test if grayscale (rank = 1, relatively)
        
        if np.linalg.matrix_rank( relative_mesh ) == 1:   # grayscale or monotone
            max_indx, min_indx = np.argmax( img_reshape ), np.argmin( img_reshape )
            
            rhs_pt, lhs_pt = img_reshape[ max_indx ], img_reshape[ min_indx ]
            mesh = np.array([ [0, 0, 0], [1, 1, 1] ])
        
            spectrum = np.linalg.norm( rhs_pt - lhs_pt )
            ratio = spectrum / ( num_colors - 1 )
            vec = ( rhs_pt - lhs_pt ) / np.linalg.norm( rhs_pt - lhs_pt )
            for i in range( num_colors - 2 ):
                new_pt = lhs_pt + ( i + 1 ) * ratio * vec 
                mesh = np.vstack( ( mesh, new_pt ) )
                
        else:
            og_hull = ConvexHull( img_reshape )
            output_rawhull_obj_file = path + "-rawconvexhull.obj"
            write_convexhull_into_obj_file( og_hull, output_rawhull_obj_file )		
            
            # get simplified convexhull (clipped already)
            mesh = simplified_convex_hull( output_rawhull_obj_file, num_colors ).vs
            
        return mesh
    
    def get_initial_weight_ls( num_colors ):
        '''
        Given: number of palette colors for the poster.
        
        Return: Initialized weight list for each color.
        '''
        
        weight_list = []
        for i in range( num_colors ):
            weight = num_colors * [0]
            weight[i] = 1
            weight_list.append( weight )
            
        return weight_list
    
    def MLO( image_og, candidate_colors, neighbor_list, weight_list):
        '''
        Given: original given image, list of candidate colors, list of 
        neighboring information, list of weights
        
        Return: result labels from MLO.
        '''
        
        # Multi-label optimization 
        print( 'start multi-label optimization...' )
        
        unary = get_unary( image_og, candidate_colors )
        binary = get_binary( neighbor_list, weight_list, candidate_colors, option = 2, penalization = penalization )
        
        # get final labels from the optimization (alpha-beta swap)
        labels = gco.cut_grid_graph_simple( unary, binary, n_iter = 100, algorithm='swap' ) 
        
        return labels
    
    def get_final_colors_from_opt_weight( labels, weight_list, palette ):
        '''
        Given: nonrepeated labels, optimized weight list, palette (numpy array).
        
        Return: a list of final colors used after MLO.
        '''
        
        # compute final colors based on our optimized weights
        final_colors = []
        for label in labels:
            w = np.array( weight_list[label] )
            color = w @ palette
            final_colors.append( list( color ) )	
            
        return final_colors
    
    def label_per_pixel_2_RGB_and_get_add_mix( labels, nonrepeated_labels, final_colors, weight_list, num_colors ):
        '''
        Given: nonrepeated labels, result labels for each pixel, final colors after MLO
        a list of weights for each selected color and number of chosen palette size.
        
        Return: RGBs per-pixel based on results labels, additive mixing layers.
        '''
        
        # convert labels to RGBs
        # visualize additive mixing layers
        image_arr = np.zeros( ( labels.size, 3 ) )
        add_mix_layers = np.zeros( ( labels.size, num_colors ) )
        for i in range( labels.size ):
            image_arr[i, :] = np.array( final_colors[ nonrepeated_labels.index( labels[i] ) ] )
            add_mix_layers[i, :] = np.array( weight_list[ labels[i] ] )
            
        # return image_arr
        return image_arr, add_mix_layers
    
    def get_RGBXY_final_colors( labels, clusters ):
        '''
        Given: labels, clusters after RGBXY clustering.
        
        Return: a list of final colors used after RGBXY clustering.
        '''
        
        # get final colors after Kmeans RGBXY
        nonrepeated_labels = get_nonrepeated_labels( labels )
        final_colors_RGBXY = []
        for label in nonrepeated_labels:
            color = clusters[label][:3]
            final_colors_RGBXY.append( color )
            
        return final_colors_RGBXY
    
    weight_list = get_initial_weight_ls( num_colors )
    palette = get_palette( input_img_path, image_arr, num_colors )
    
    candidate_colors, neighbor_list, weight_list = \
    get_candidate_colors_and_neighbor_list( palette, weight_list, num_colors, num_blend ) 
    
    # MLO
    mlo_labels = MLO( image_og, candidate_colors, neighbor_list, weight_list )
    
    # locally optimize weights
    nonrepeated_mlo_labels = get_nonrepeated_labels( mlo_labels )
    optimized_weight_list = optimize_weight( mlo_labels, nonrepeated_mlo_labels, weight_list, palette, image_arr )
    
    # compute final optimized colors in each layer
    mlo_final_colors = get_final_colors_from_opt_weight( nonrepeated_mlo_labels, optimized_weight_list, palette )
    print( 'Final colors being used after MLO:', len( mlo_final_colors ) ) 
    print( 'Posterization Done! Extract mixing layers...')
    
    # reconstruct per-pixel label to per-pixel RGB 
    image_mlo, add_mix_layers = label_per_pixel_2_RGB_and_get_add_mix( mlo_labels, nonrepeated_mlo_labels\
        , mlo_final_colors, optimized_weight_list, num_colors )
    print( 'Done extracting layers!' )
    
    image_RGB = image_mlo.reshape( image_og.shape )
    
    return image_RGB, mlo_final_colors, add_mix_layers, palette


######################################################
######################################################
######################################################


def post_smoothing( img_mlo, threshold, blur_window = 7, blur_map = None ):
    
    print( 'Start smoothing images... ')
    
    def medianBlur_truncate( img, dft_img, window_size ):
        filtered_img = cv2.medianBlur( img, window_size )
        if blur_map is None:
            filtered_img[ dft_img >= threshold ] = img[ dft_img >= threshold ]
        else:
            filtered_img[ dft_img >= blur_map ] = img[ dft_img >= blur_map ]
        return filtered_img
    
    cv2_img_mlo = np.array( img_mlo )
    
    # Convert RGB to BGR 
    cv2_img_mlo = cv2_img_mlo[:, :, ::-1].copy()    # in cv2 format
    
    img_mlo_og = np.copy( cv2_img_mlo )
    gray = cv2.cvtColor( cv2_img_mlo, cv2.COLOR_BGR2GRAY )
    
    # DFT and inverse DFT
    rows, cols = gray.shape
    crow, ccol = int( rows / 2 ) , int( cols / 2 )
    f = np.fft.fft2( gray )
    fshift = np.fft.fftshift( f )
    
    fshift[crow - 25: crow + 25, ccol - 25: ccol + 25] = 0
    f_ishift = np.fft.ifftshift( fshift )
    img_back = np.fft.ifft2( f_ishift )
    img_back = np.abs( img_back ) / 255.    # black: 0, white: 1
    
    cv2_img_mlo = medianBlur_truncate( cv2_img_mlo, img_back, blur_window )
    cv2_img_mlo = medianBlur_truncate( cv2_img_mlo, img_back, blur_window )
    cv2_img_mlo = medianBlur_truncate( cv2_img_mlo, img_back, blur_window )
    
    cv2_img_mlo = cv2.cvtColor( cv2_img_mlo, cv2.COLOR_BGR2RGB)
    
    print( 'Image smoothing Done!' )
    
    return cv2_img_mlo
