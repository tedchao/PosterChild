import sys
import numpy as np
from PIL import Image

from skimage import color
from sklearn.cluster import KMeans
import cv2

from math import *

# import potrace
import cairo
import random
import time

import simplepalettes
from .simplify_convexhull import *

def get_candidate_colors_and_neighbor_list( mesh, weight_list, num_colors, num_blend ):
    '''
    Given: palette (np array of colors), list of weights for each color (2D list),
    number of colors, number of blends.
    
    Return: A neighboring information for different blends and palette. (2D list),
    list of weights for each color (2D list), a list of candidate colors.
    '''
    
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
            
    return candidate_colors, neighbor_list, np.array( weight_list )
    

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
    
    def frobenius_inner_product( matrix1, matrix2 ):
        '''
        Given: Two matrices.
        
        Return: Frobenius inner product of two matrices.
        '''
        assert matrix1.shape == matrix2.shape
        return np.trace( matrix1.T @ matrix2 )
        
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
        import gco
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
            image_arr[i, :] = np.array( final_colors[nonrepeated_labels.index( labels[i] )] )
            add_mix_layers[i, :] = np.array( weight_list[nonrepeated_labels.index( labels[i] )] )
        
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
    
    
    # reconstruct per-pixel label to per-pixel RGB 
    image_mlo, add_mix_layers = \
    label_per_pixel_2_RGB_and_get_add_mix( mlo_labels, nonrepeated_mlo_labels, mlo_final_colors, optimized_weight_list, num_colors )
    
    # reduce similar color layers using clustering in RGBXY
    image_RGBXY, RGBXY_labels, RGBXY_clusters = \
    get_kmeans_cluster_image( 0, image_mlo, width, height, mlo_final_colors, optimized_weight_list, option = 2 )
    
    final_colors_RGBXY = get_RGBXY_final_colors( RGBXY_labels, RGBXY_clusters )
    
    return image_RGBXY, final_colors_RGBXY, add_mix_layers, palette.vs 
    

def main():
    import argparse
    parser = argparse.ArgumentParser( description = 'Posterization.' )
    parser.add_argument( 'input_image', help = 'The path to the input image.' )
    parser.add_argument( 'output_posterized_path', help = 'Where to save the output posterized image.' )
    parser.add_argument( 'output_add_mix_path', help = 'The path to the output additive-mixing image.' )
    args = parser.parse_args()

    img_arr = np.asfarray( Image.open(args.input_image).convert( 'RGB' ) ) / 255. 
    img_arr_og = np.copy( img_arr )
    
    # algorithm starts
    start = time.time()
    
    # get Kmeans clustered image from RGB space
    num_clusters = 20
    img_arr_re = img_arr.reshape( ( -1, 3 ) )
    img_arr_cluster = get_kmeans_cluster_image( num_clusters, img_arr_re, img_arr.shape[0], img_arr.shape[1] )
    
    # get posterized image
    palette_num = 6
    num_blend = 3  
    post_img, final_colors, add_mix_layers, palette = \
    posterization( args.input_image, img_arr_og, img_arr_cluster, palette_num, num_blend )
    
    Image.fromarray( np.clip( 0, 255, post_img*255. ).astype( np.uint8 ), 'RGB' ).save( args.output_posterized_path + '.jpg' )
    
    # save palette and final used colors
    timg = np.clip( 0, 255, simplepalettes.palette2swatch( palette ) * 255. ).astype( np.uint8 )
    final_colors = np.array( final_colors )
    simplepalettes.save_image_to_file( timg, args.output_posterized_path + '-' + 'palette.png', clobber = True )
    simplepalettes.save_palette2wedges( final_colors, args.output_posterized_path + '-' + 'all_colors.png', clobber = True )
    
    # save additive mixing layers
    save_additive_mixing_layers( add_mix_layers, img_arr.shape[0], img_arr.shape[1], palette, args.output_add_mix_path )

    end = time.time()
    print( "Finished. Total time: ", end - start )
    
if __name__ == '__main__':
    main()