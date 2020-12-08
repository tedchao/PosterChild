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

import simplepalettes
from simplify_convexhull import *

from tkinter import *
import tkinter.filedialog
import tkinter.messagebox


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
        neighbor_list.append( mesh.vertex_vertex_neighbors ( i ) )
    
    # discrete blendings
    candidate_colors = mesh.vs
    pos_iter = num_colors
        
    for i in range( num_colors ):
        for j in range( i+1, num_colors ):
            
            # add blended colors from i-position to j-position
            for k in range( num_blend ):
                candidate_colors = np.vstack( ( candidate_colors, ( 1 - ( k + 1 ) * size ) * candidate_colors[i] + ( k + 1 ) *  size * candidate_colors[j] ) ) 
                
                if k == 0 and k == num_blend - 1:
                    neighbor_list.append( [i, j] )  # only 1 blend
                elif k == 0 or k == num_blend - 1:
                    if k == 0:
                        neighbor_list.append( [i, pos_iter + 1] )   # first head
                    else:
                        neighbor_list.append( [j, pos_iter + num_blend - 2] )   # last tail
                else:
                    #if k != 0 and k != num_blend - 1:
                    neighbor_list.append( [pos_iter + k - 1, pos_iter + k + 1] )
            
            if i in neighbor_list[j]:
                neighbor_list[i].remove( j )
                neighbor_list[j].remove( i )
                
            # add palette weight for each color to weight list
            for s in range( num_blend ):
                weights = num_colors * [0]
                weights[i] = 1 - ( s + 1 ) * size
                weights[j] = ( s + 1 ) *  size
                weight_list.append( weights )
                
            pos_iter += num_blend
            
            
            '''
            if num_blend == 3:
                # add to candidate colors
                candidate_colors = np.vstack( ( candidate_colors, a * candidate_colors[i] + (1-a) * candidate_colors[j] ) ) # middle blend
                candidate_colors = np.vstack( ( candidate_colors, b * candidate_colors[i] + (1-b) * candidate_colors[j] ) ) # close to j
                candidate_colors = np.vstack( ( candidate_colors, c * candidate_colors[i] + (1-c) * candidate_colors[j] ) ) # close to i
                    
                neighbor_list[i].append( pos_iter + 2 ) 
                neighbor_list[j].append( pos_iter + 1 )
                    
                neighbor_list.append( [pos_iter+1, pos_iter+2] )     # first color
                neighbor_list.append( [j, pos_iter] )
                neighbor_list.append( [i, pos_iter] )
                    
                if i in neighbor_list[j]:
                    neighbor_list[i].remove( j )
                    neighbor_list[j].remove( i )
                    
                # add palette weight for each color to weight list
                for s in range( 3 ):
                    weights = num_colors * [0]
                    weights[i] = blends[s][0]
                    weights[j] = blends[s][1]
                    weight_list.append( weights )
                    
                pos_iter += 3
                
            if num_blend == 1:
                
                # add to candidate colors
                candidate_colors = np.vstack( ( candidate_colors, a * candidate_colors[i] + ( 1 - a ) * candidate_colors[j] ) ) # middle blend
                
                neighbor_list[i].append( pos_iter )
                neighbor_list[j].append( pos_iter )
                
                neighbor_list.append( [i, j] )
                
                if i in neighbor_list[j]:
                    neighbor_list[i].remove( j )
                    neighbor_list[j].remove( i )
                
                weights = num_colors * [0]
                weights[i] = blends[0][0]
                weights[j] = blends[0][1]
                weight_list.append( weights )
                pos_iter += 1
            '''
            
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


def get_binary( neighbor_list, weight_list, candidate_colors, option = 1 ):
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
                    binary[i][j] = np.linalg.norm( ( candidate_colors[i] - candidate_colors[j] ), ord=2 )
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


def posterization( input_img_path, image_og, image_arr, num_colors, num_blend = 3 ):
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
        
        Return: Simplified convexhull for clustered input image (mesh object).
        '''
        
        # reshape image array into scipy.convexhull
        img_reshape = image_arr.reshape( ( -1, 3 ) )
        og_hull = ConvexHull( img_reshape )
        output_rawhull_obj_file = path + "-rawconvexhull.obj"
        write_convexhull_into_obj_file( og_hull, output_rawhull_obj_file )		
        
        # get simplified convexhull (clipped already)
        mesh = simplified_convex_hull( output_rawhull_obj_file, num_colors )
        
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
        binary = get_binary( neighbor_list, weight_list, candidate_colors, option = 2 )
            
        # get final labels from the optimization (alpha-beta swap)
        labels = gco.cut_grid_graph_simple( unary, binary, n_iter = 100, algorithm='swap' ) 
        
        return labels
    
    def get_final_colors_from_opt_weight( labels, weight_list, palette ):
        '''
        Given: nonrepeated labels, optimized weight list.
        
        Return: a list of final colors used after MLO.
        '''
        
        # compute final colors based on our optimized weights
        final_colors = []
        for label in labels:
            w = np.array( weight_list[label] )
            color = w @ palette.vs
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
    
    
    # get blended colors, neighbor list and weight list
    weight_list = get_initial_weight_ls( num_colors )
    palette = get_palette( input_img_path, image_arr, num_colors )
    
    candidate_colors, neighbor_list, weight_list = \
    get_candidate_colors_and_neighbor_list( palette, weight_list, num_colors, num_blend ) 
    
    
    # MLO
    mlo_labels = MLO( image_og, candidate_colors, neighbor_list, weight_list )
    
    # locally optimize weights
    nonrepeated_mlo_labels = get_nonrepeated_labels( mlo_labels )
    optimized_weight_list = optimize_weight( mlo_labels, nonrepeated_mlo_labels, weight_list, palette.vs, image_arr )
    
    # compute final optimized colors in each layer
    mlo_final_colors = get_final_colors_from_opt_weight( nonrepeated_mlo_labels, optimized_weight_list, palette )
    print( 'Final colors being used after MLO:', len( mlo_final_colors ) ) 
    print( 'Posterization Done! Extract mixing layers...')
    
    # reconstruct per-pixel label to per-pixel RGB 
    image_mlo, add_mix_layers = label_per_pixel_2_RGB_and_get_add_mix( mlo_labels, nonrepeated_mlo_labels\
        , mlo_final_colors, optimized_weight_list, num_colors )
    print( 'Done extracting layers!' )
    
    image_RGB = image_mlo.reshape( image_og.shape )
    
    return image_RGB, mlo_final_colors, add_mix_layers, palette.vs 


######################################################
######################################################
######################################################

def post_smoothing( img_mlo, threshold ):
    
    print( 'Start smoothing images... ')
    
    @jit(nopython=True)
    def get_median_color( window ):
        
        channel1 = window[:, :, 0]
        channel2 = window[:, :, 1]
        channel3 = window[:, :, 2]
        
        color = np.zeros( (  3 ), dtype = np.uint8 )
        
        color[0] = np.median( channel1 )
        color[1] = np.median( channel2 )
        color[2] = np.median( channel3 )
        
        return color
    
    @jit(nopython=True)
    def medianBlur_truncate( img, dft_img, window_size ):
        margin = int( ( window_size - 1 ) / 2 )
        
        filtered_img = np.copy( img )
        for i in range( margin, img.shape[0] - margin ):
            for j in range( margin, img.shape[1] - margin ):
                
                if dft_img[i, j] < threshold:
                    window = img[i - margin: i + margin + 1, j - margin: j + margin + 1]
                    
                    median_color = get_median_color( window )
                    filtered_img[i, j] = median_color
                    
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

    cv2_img_mlo = medianBlur_truncate( cv2_img_mlo, img_back, 7 )
    cv2_img_mlo = medianBlur_truncate( cv2_img_mlo, img_back, 7 )
    cv2_img_mlo = medianBlur_truncate( cv2_img_mlo, img_back, 7 )
    
    cv2_img_mlo = cv2.cvtColor( cv2_img_mlo, cv2.COLOR_BGR2RGB)
    
    print( 'Image smoothing Done!' )
    
    return PIL.Image.fromarray( cv2_img_mlo )
    
    

def posterized_pipline( path, img_arr, img_og, threshold = 0.1, num_clusters = 20, num_blend = 3, palette_num = 6 ):
    global tk_posterized_image, tk_palette_color, tk_add_mix, tk_img_shape
    
    # algorithm starts
    start = time.time()
    
    # K-means
    img_arr_re = img_arr.reshape( ( -1, 3 ) )
    img_arr_cluster = get_kmeans_cluster_image( num_clusters, img_arr_re, img_arr.shape[0], img_arr.shape[1] )
    
    # MLO
    post_img, final_colors, add_mix_layers, palette = \
    posterization( path, img_og, img_arr_cluster, palette_num, num_blend )
    tk_img_shape = post_img.shape
    
    # convert to uint8 format
    post_img = PIL.Image.fromarray( np.clip( 0, 255, post_img*255. ).astype( np.uint8 ), 'RGB' )
    tk_posterized_image = post_img
    tk_palette_color = palette
    tk_add_mix = add_mix_layers
    
    # post-smoothing
    # smooth_post_img = post_smoothing( post_img, threshold )
    
    end = time.time()
    print( "Finished. Total time: ", end - start )
    
    return post_img
    
    
def select_image():
    global panel, path, tk_input_image, tk_switch, tk_posterized_image
    global tk_num_clusters, tk_palette_size, tk_num_blend, tk_thres
    global tk_pal_num, tk_rc_r, tk_rc_g, tk_rc_b
    
    # assignment on global variables
    path = tkinter.filedialog.askopenfilename()
    tk_input_image = PIL.Image.open( path ).convert( 'RGB' )    # pillow image
    
    # convert Pillow image to ImageTK
    tk_image = PIL.ImageTk.PhotoImage( tk_input_image )
    
    if panel is None:
        panel = Label( image = tk_image )
        panel.image = tk_image
        panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)
        
    else:
        tk_posterized_image = None
        panel.configure( image = tk_image )
        panel.image = tk_image

    c_kms = Label( root, text = '# of clusters for K-means (default: 20): ')
    tk_num_clusters = Entry(root)
    c_kms.grid(row=20, column=0)
    tk_num_clusters.grid(row=21, column=0)
    
    p_sz = Label( root, text = 'Palette size (default: 6): ')
    tk_palette_size = Entry(root)
    p_sz.grid(row=22, column=0)
    tk_palette_size.grid(row=23, column=0)
    
    n_b = Label( root, text = 'Numbers of blending ways: ')
    tk_num_blend = Entry(root)
    n_b.grid(row=24, column=0)
    tk_num_blend.grid(row=25, column=0)
    
    thres = Label( root, text = 'Blurring threshold (0 to 1, default: 0.1): ')
    tk_thres = Entry(root)
    thres.grid(row=26, column=0)
    tk_thres.grid(row=27, column=0)
    
    pal_num = Label( root, text = 'Choose palette number to recolor (1 to palette size): ')
    tk_pal_num = Entry(root)
    pal_num.grid(row=50, column=0)
    tk_pal_num.grid(row=51, column=0)
    
    recolor = Label( root, text = 'Choose color to recolor (0 to 255): ')
    tk_rc_r = Entry(root)
    tk_rc_g = Entry(root)
    tk_rc_b = Entry(root)
    recolor.grid(row=60, column=0)
    tk_rc_r.grid(row=61, column=0)
    tk_rc_g.grid(row=62, column=0)
    tk_rc_b.grid(row=63, column=0)

    
    
def posterize_button():
    global tk_posterized_image
    
    if panel is None:
        tkinter.messagebox.showwarning(title='Warning', message='Please select an image first.')
    
    else:
        # true image to work on
        img_arr = np.asfarray( PIL.Image.open( path ).convert( 'RGB' ) ) / 255. 
        img_arr_og = np.copy( img_arr )
    
        
        if tk_num_clusters.get():
            num_clusters = int( tk_num_clusters.get() )
        else:
            num_clusters = 20
        
        if tk_palette_size.get():
            palette_size = int( tk_palette_size.get() )
        else:
            palette_size = 6
        
        if tk_num_blend.get():
            num_blend = int( tk_num_blend.get() )
        else:
            num_blend = 3
            
        if tk_thres.get():
            threshold = float( tk_thres.get() )
        else:
            threshold = 0.1
        
        posterized_image = posterized_pipline( path, img_arr, img_arr_og, threshold, num_clusters, num_blend, palette_size )
        
        tk_switch = 1
        tk_posterized_image = posterized_image
        panel.image.paste( posterized_image )
        panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)
        print( 'palette colors: ', np.clip( 0, 255, tk_palette_color * 255. ).astype( np.uint8 ) )


def smooth_image():
    global tk_posterized_image, smooth_posterized_image
    
    if panel is None:
        tkinter.messagebox.showwarning( title='Warning', message='Please select an image first.' )
        
    else:
        if tk_posterized_image is None:
            tkinter.messagebox.showwarning( title='Warning', message='Please posterize the image before smoothing the bouandries.' )
            
        else:
            if not tk_thres.get():
                tkinter.messagebox.showwarning( title='Warning', message='Please specify blurring threshold.' )
            
            else:
                smooth_posterized_image = post_smoothing( tk_posterized_image, float( tk_thres.get() ) )
                panel.image.paste( smooth_posterized_image )
                panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)

def compare():
    global tk_switch
    
    if panel is None:
        tkinter.messagebox.showwarning( title='Warning', message='Please select an image first.' )
        
    else:
        if tk_posterized_image is None:
            tkinter.messagebox.showwarning( title='Warning', message='Please posterize the image before comparing with the original image.' )
        
        elif smooth_posterized_image is None:
            tkinter.messagebox.showwarning( title='Warning', message='Please smooth the posterized image before comparing with the original image.' )
            
        else:
            if tk_switch == 0:
                panel.image.paste( tk_input_image )
                panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)
                tk_switch = 1
            else:
                panel.image.paste( tk_posterized_image )
                panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)
                tk_switch = 0


def savefile():
    
    if panel is None:
        tkinter.messagebox.showwarning( title='Warning', message='Please select an image first.' )
        
    else:
        if tk_posterized_image is None:
            tkinter.messagebox.showwarning(title='Warning', message='Please posterize the image before saving it.')
        
        elif smooth_posterized_image is None:
            tkinter.messagebox.showwarning( title='Warning', message='Please smooth the posterized image before saving it.' )
        
        else:
            filename = tkinter.filedialog.asksaveasfilename( defaultextension=".png" )
            if not filename:
                return
            smooth_posterized_image.save( filename )


def recolor_posterized_image():
    global tk_recolor_image
    
    if panel is None:
        tkinter.messagebox.showwarning( title='Warning', message='Please select an image first.' )
        
    else:
        if tk_posterized_image is None:
            tkinter.messagebox.showwarning(title='Warning', message='Please posterize the image before recoloring it.')
        else:
            if tk_pal_num.get():
                chosen_palette = int( tk_pal_num.get() ) - 1
            else:
                chosen_palette = 0

            if tk_rc_r.get() == '' or tk_rc_g.get() == '' or tk_rc_b.get() == '':
                tkinter.messagebox.showwarning(title='Warning', message='Please recolor with a valid color in RGB.')
            
            else:
                recolor_paint = np.array( [int( tk_rc_r.get() ), int( tk_rc_g.get() ), int( tk_rc_b.get() ) ] ) / 255.
                tk_palette_color[ chosen_palette ] = recolor_paint
                
                recolor_image = tk_add_mix @ tk_palette_color
                recolor_image = recolor_image.reshape( tk_img_shape )
                
                recolor_image = PIL.Image.fromarray( np.clip( 0, 255, recolor_image*255. ).astype( np.uint8 ), 'RGB' )
                
                if tk_thres.get():
                    threshold = float( tk_thres.get() )
                else:
                    threshold = 0.1
                    
                smooth_recolor_img = post_smoothing( recolor_image, threshold )
                tk_recolor_image = smooth_recolor_img
                
                panel.image.paste( smooth_recolor_img )
                panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)
            

def save_recolor():
    
    if panel is None:
        tkinter.messagebox.showwarning( title='Warning', message='Please select an image first.' )
        
    else:
        if tk_recolor_image is None:
            tkinter.messagebox.showwarning(title='Warning', message='Please recolor the image before saving it.')
        else:
            filename = tkinter.filedialog.asksaveasfilename( defaultextension=".png" )
            if not filename:
                return
            tk_recolor_image.save( filename )


root = Tk()
root.title( 'Posterization' )
    
panel = None
tk_switch = 0
tk_posterized_image = None
tk_recolor_image = None
smooth_posterized_image = None
'''
btn1 = Button(root, text="Select an image", command = select_image)
btn1.pack(side="bottom", fill="both", expand="yes")

btn2 = Button(root, text="Posterized!", command = posterize_button)
btn2.pack(side="bottom", fill="both", expand="yes")

btn3 = Button(root, text="Press to compare", command = compare)
btn3.pack(side="bottom", fill="both", expand="yes")

btn4 = Button(root, text="Press to save posterized image", command=savefile)
btn4.pack(side="bottom", fill="both", expand="yes")
'''

f1 = Frame(root)
btn1 = Button(f1, text="Select an image", command = select_image).pack(side=TOP, fill="both", expand="yes")
btn2 = Button(f1, text="Posterize!", command = posterize_button).pack(side=TOP, fill="both", expand="yes")
btn3 = Button(f1, text="Smooth!", command = smooth_image).pack(side=TOP, fill="both", expand="yes")
btn4 = Button(f1, text="Press to compare", command = compare).pack(side=TOP, fill="both", expand="yes")
btn5 = Button(f1, text="Save posterized image", command = savefile).pack(side=TOP, fill="both", expand="yes")
btn6 = Button(f1, text="Recolor posterized image", command = recolor_posterized_image).pack(side=TOP, fill="both", expand="yes")
btn7 = Button(f1, text="Save recolored image", command = save_recolor).pack(side=TOP, fill="both", expand="yes")
f1.grid(row=0, column=0)

# kick off the GUI
root.mainloop()
    
    
    