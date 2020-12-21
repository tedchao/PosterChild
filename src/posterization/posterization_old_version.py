import sys
import numpy as np
from PIL import Image

from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import coo_matrix
from scipy.optimize import *
from skimage import color
from skimage.morphology import binary_opening
from sklearn.cluster import KMeans

from math import *
from .trimesh import TriMesh

import cvxopt
import potrace
import cairo
import random
import time

from . import simplepalettes
from . import extract_brush


# Vectorzied a raster image
class Vectorized_image( object ):
    def __init__( self, filename, width, height, boundaries, final_colors ):
        self.surface = cairo.SVGSurface( filename + '.svg', height, width )
        cr = cairo.Context( self.surface )
        self.cr = cr
        
        # set up background
        cr.scale( height, width )
        cr.set_line_width( 1 )

        cr.rectangle( 0, 0, 1, 1 )
        cr.set_source_rgb( 1, 1, 1 ) # white
        cr.fill()
        
        self.draw_dest( cr, height, width, boundaries, final_colors )
        
        self.surface.write_to_png( filename + '.png' )
        cr.show_page()
        self.surface.finish()
    
    def draw_dest( self, cr, height, width, boundaries, final_colors ):
        print( "Start vectorizing images..." )

        #print(len(final_colors))
        
        # Iterate over path curves
        for i in range( len( boundaries ) ):
            painted = []
            for curve in boundaries[i]:
                #print ("start_point =", curve.start_point)
                # draw parent curve
                if curve not in painted:
                    cr.move_to( curve.start_point[0] / height, curve.start_point[1] / width )
                    for segment in curve:
                        #print (segment)
                        end_point_x, end_point_y = segment.end_point
                        
                        if segment.is_corner:
                            c_x, c_y = segment.c
                            cr.line_to( c_x / height, c_y / width )
                        else:
                            c1_x, c1_y = segment.c1
                            c2_x, c2_y = segment.c2
                            cr.curve_to( c1_x / height, c1_y / width, c2_x / height, c2_y / width, \
                            end_point_x / height, end_point_y / width)
                    cr.close_path()
                
                # if the parent have children, then keep drawing children
                if curve.children:
                    for child in curve.children:
                        painted.append(child)
                        cr.move_to(child.start_point[0]/height, child.start_point[1]/width)
                        for segment in child:
                            #print (segment)
                            end_point_x, end_point_y = segment.end_point
                            
                            if segment.is_corner:
                                c_x, c_y = segment.c
                                cr.line_to(c_x/height, c_y/width)
                            else:
                                c1_x, c1_y = segment.c1
                                c2_x, c2_y = segment.c2
                                cr.curve_to(c1_x/height, c1_y/width, c2_x/height, c2_y/width, \
                                end_point_x/height, end_point_y/width)
                        cr.close_path()
                                               
                #cr.line_to(curve.start_point[0]/height, curve.start_point[1]/width)
                cr.set_source_rgba(final_colors[i][0], final_colors[i][1], final_colors[i][2], 1)
                
                # This means that any area that is inside an odd number of subpaths will be filled
                #, any area that is inside an even number of subpaths will be unfilled.
                cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
                cr.fill()

# Trace the boundaries of different color regions
def get_boundaries(labels, unoverlap_labels, img_w, img_h):
    
    boundaries = []
    
    num_label = len(unoverlap_labels)
    
    #unoverlap_labels = get_drawing_order(labels, unoverlap_labels) 
    layer = np.ones([img_w, img_h])
    # loop through all the labels and convert each label into a bitmap in each step
    for label in unoverlap_labels:
        
        # Create a bitmap from the array
        bmp = potrace.Bitmap(layer)

        # Trace the bitmap
        path = bmp.trace(alphamax=1.3)
        boundaries.append(path)
        
        spec_label = np.copy(labels)
        for i in range(spec_label.size):
            if spec_label[i] == label:   # if the pixel == label, we mark it
                spec_label[i] = 1
            else:						 # else: we do not trace it!
                spec_label[i] = 0
        spec_label_pos = spec_label.reshape(img_w, img_h)
        layer -= spec_label_pos
        
    return boundaries


def get_candidate_colors_and_neighbor_list(mesh, weight_list, num_colors, num_blend):
    
    a, b, c = 0.5, 0.25, 0.75
    blends = [[a, 1-a], [b, 1-b], [c, 1-c]]
    
    # store the neighbors
    # 2D list
    neighbor_list = []
    for i in range(num_colors):
        neighbor_list.append(mesh.vertex_vertex_neighbors(i))
    
    # discrete blendings
    candidate_colors = mesh.vs
    pos_iter = num_colors
        
    for i in range(num_colors):
        for j in range(i+1, num_colors):
            # add to candidate colors
            candidate_colors = np.vstack((candidate_colors, a*candidate_colors[i] + (1-a)*candidate_colors[j])) # middle blend
            candidate_colors = np.vstack((candidate_colors, b*candidate_colors[i] + (1-b)*candidate_colors[j])) # close to j
            candidate_colors = np.vstack((candidate_colors, c*candidate_colors[i] + (1-c)*candidate_colors[j])) # close to i
                
            neighbor_list[i].append(pos_iter + 2)
            neighbor_list[j].append(pos_iter + 1)
                
            neighbor_list.append([pos_iter+1, pos_iter+2])     # first color
            neighbor_list.append([j, pos_iter])
            neighbor_list.append([i, pos_iter])
                
            if i in neighbor_list[j]:
                neighbor_list[i].remove(j)
                neighbor_list[j].remove(i)
                
            # add palette weight for each color to weight list
            for s in range(3):
                weights = num_colors * [0]
                weights[i] = blends[s][0]
                weights[j] = blends[s][1]
                weight_list.append(weights)
                
            pos_iter += 3
            
    return candidate_colors, neighbor_list, np.array(weight_list)


# Unary term in minimizing energy function
def get_unary(image_arr, candidate_colors):
    
    # unary.shape = (486, 864, len(labels)) ideally!
    unary = np.zeros((image_arr.shape[0], image_arr.shape[1], len(candidate_colors)))
        
    for i in range(len(candidate_colors)):
        dist_square = (candidate_colors[i] - image_arr) ** 2
        unary[:, :, i] = np.sqrt(np.sum(dist_square, axis = 2))
    return unary
    

# Binary term in minimizing energy function
def get_binary(neighbor_list, weight_list, candidate_colors, regularization, option=1):

    # 1 if neighbor, (inf) if not neighbor, diag(binary) = 0
    if option == 1:
        binary = np.ones((len(neighbor_list), len(neighbor_list))) * np.inf
        
        for i in range(len(candidate_colors)):
            for j in range(len(candidate_colors)):
                if i == j:
                    binary[i][j] = 0
        
        for i in range(len(neighbor_list)):
            for j in range(len(neighbor_list[i])):
                binary[i][neighbor_list[i][j]] = 0.5
    
    # half of the L1 norm distance based on weights on each palette
    if option == 2:
        binary = np.zeros((len(neighbor_list), len(neighbor_list)))
        
        for i in range(len(candidate_colors)):
            for j in range(len(candidate_colors)):
                if i != j:
                    #binary[i][j] = 0.3 * np.linalg.norm((weight_list[i] - weight_list[j]), ord=1)
                    binary[i][j] = np.linalg.norm((candidate_colors[i] - candidate_colors[j]), ord=2)
    #print(binary)
    return binary


# Get nonrepeated labels
def get_unoverlap_labels(colors):
    # option 1 means list
    visited = []
    for label in colors:
        if label not in visited:
            visited.append(label)
    return visited


def frobenius_inner_product(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    return np.trace(matrix1.T @ matrix2)


def optimize_weight(labels, unoverlap_labels, weight_list, palette, image_arr):
    width = image_arr.shape[0]
    height = image_arr.shape[1]
    
    labels = np.asfarray(labels).reshape(width, height, 1)
    palette_labels = []
    for i in range(len(palette)):
        palette_labels.append(i)
    
    for label in unoverlap_labels:
        if label not in palette_labels:
            F_label = []
            for i in range(width):
                for j in range(height):
                    if labels[i][j] == label:
                        F_label.append(image_arr[i, j, :])
            
            # quadratic programming            
            F_prime = np.array(F_label)
            pixel_size = F_prime.shape[0]           
            
            w_lst = weight_list[label]
            
            Pi_color = palette[np.nonzero(w_lst)[0][0]][np.newaxis]    # shape:1x3
            Pj_color = palette[np.nonzero(w_lst)[0][1]][np.newaxis]
            
            Is = np.ones(pixel_size)[np.newaxis]
            Pi = Is.T @ Pi_color
            Pj = Is.T @ Pj_color
            
            # some coefficients
            Pi_Pi = frobenius_inner_product(Pi, Pi)
            Pj_Pj = frobenius_inner_product(Pj, Pj)
            Pi_Pj = frobenius_inner_product(Pi, Pj)
            F_Pi = frobenius_inner_product(F_prime, Pi)
            F_Pj = frobenius_inner_product(F_prime, Pj)
            
            a = frobenius_inner_product(Pi - Pj, Pi - Pj)
            b = 2 * (Pi_Pj - F_Pi + F_Pj - Pj_Pj)
            
            x = max(0, min(1, -b/(2*a)))
            
            nonzero_pos = np.nonzero(w_lst)[0]
            weight_list[label][nonzero_pos[0]] = x
            weight_list[label][nonzero_pos[1]] = 1-x
    
    return weight_list


# perform clustering
def k_means_cluster(num_clusters, post_img, width, height):
    # input: post_img has shape (width x height, 3)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(post_img)
    kmeans_labels = kmeans.labels_
    clusters = kmeans.cluster_centers_

    for i in range(kmeans_labels.size):
        post_img[i, :] = clusters[kmeans_labels[i]]
    post_img = np.asfarray(post_img).reshape(width, height, 3)
    
    return post_img

def convertRGBXY(img, width, height):
    # input: img has shape (width x height, 3)
    img_rgbxy = np.zeros((img.shape[0], 5))
    img = np.asfarray(img).reshape(width, height, 3)
    img_rgbxy = np.asfarray(img_rgbxy).reshape(width, height, 5)
    
    for i in range(width):
        for j in range(height):
            img_rgbxy[i, j, :3] = img[i, j]
            img_rgbxy[i, j, 3:5] = (1/600) * np.array([i, j])
    img_rgbxy = img_rgbxy.reshape((-1,5))
    
    return img_rgbxy


# look at the weight list to see the nonzero index are the same
def get_num_clusters(weight_ls):
    nonzero_indices = []
    for weights in weight_ls:
        w_nonzero = [i for i, e in enumerate(weights) if e != 0]
        if w_nonzero not in nonzero_indices:
            nonzero_indices.append(w_nonzero)
    return len(nonzero_indices)



def boundries_smoothing( unoverlap_labels, labels ):
    pass




def posterization( path, img_og, image_arr, num_colors, num_blend = 1 ):
    '''
    Given:
        image_arr: An n-rows by m-columns array of RGB colors.
        num_colors: number of ink colors for the poster.
    '''
    assert len( image_arr.shape ) == 3
    assert num_colors == int( num_colors ) and num_colors > 0
    
    width = image_arr.shape[0]
    height = image_arr.shape[1]
    
    # reshape image array into scipy.convexhull
    img_reshape = image_arr.reshape( ( -1, 3 ) )
    og_hull = ConvexHull( img_reshape )
    output_rawhull_obj_file = path + "-rawconvexhull.obj"
    write_convexhull_into_obj_file( og_hull, output_rawhull_obj_file )		
    
    # get simplified convexhull
    # clipped already
    mesh = simplified_convex_hull( output_rawhull_obj_file, num_colors )

    # initialize weight list
    weight_list = []
    for i in range( num_colors ):
        weight = num_colors * [0]
        weight[i] = 1
        weight_list.append( weight )
    
    
    # get blended colors and neighbor list
    num_blend = 3
    candidate_colors, neighbor_list, weight_list_sep = \
    get_candidate_colors_and_neighbor_list( mesh, weight_list, num_colors, num_blend ) 
    
    
    # Multi-label optimization 
    import gco
    print( 'start multi-label optimization...' )
    unary = get_unary( img_og, candidate_colors )
    binary = get_binary( neighbor_list, weight_list_sep, candidate_colors, 0.1, 2 )
        
    # get final labels from the optimization	
    labels = gco.cut_grid_graph_simple( unary, binary, n_iter = 100, algorithm='swap' ) # alpha-beta-swap

    # locally optimize weights
    unoverlap_labels = get_unoverlap_labels( labels )
    weight_list_sep = optimize_weight( labels, unoverlap_labels, weight_list_sep, mesh.vs, image_arr )
    
    
    # compute final colors based on our optimized weights
    final_colors = []
    for label in unoverlap_labels:
        w = np.array( weight_list_sep[label] )
        color = w @ mesh.vs
        final_colors.append( list( color ) )	
    
    print( 'Final colors being used after MLO:', len( final_colors ) ) 
    
    # convert labels to RGBs
    # visualize additive mixing layers
    post_img = np.zeros( ( labels.size, 3 ) )
    add_mix_layers = np.zeros( ( labels.size, num_colors ) )
    for i in range( labels.size ):
        post_img[i, :] = np.array( final_colors[unoverlap_labels.index( labels[i] )] )
        add_mix_layers[i, :] = np.array( weight_list_sep[unoverlap_labels.index( labels[i] )] )
    
    # reconstruct segmented images
    #post_img = np.asfarray(post_img).reshape(width, height, 3)
    
    
    # K-means on RGBXY
    num_cluster = get_num_clusters( weight_list_sep )
    if num_cluster < len( final_colors ) - 10:
        num_cluster = len( final_colors ) - 10
    
    img_rgbxy = convertRGBXY( post_img, width, height )
    kmeans = KMeans( n_clusters=num_cluster, random_state=0 ).fit( img_rgbxy )
    kmeans_labels = kmeans.labels_
    clusters = kmeans.cluster_centers_
    
    print( 'Final colors being used after Kmeans RGBXY:', clusters.shape[0] )
    
    post_img = np.zeros( ( labels.size, 3 ) )
    for i in range( kmeans_labels.size ):
        color = clusters[kmeans_labels[i]][:3]
        post_img[i, :] = color
    post_img = np.asfarray( post_img ).reshape( width, height, 3 )
    
    # get final colors after Kmeans RGBXY
    unoverlap_labels_kmeans = get_unoverlap_labels( kmeans_labels )
    final_colors_Kmeans = []
    for label in unoverlap_labels_kmeans:
        color = clusters[label][:3]
        final_colors_Kmeans.append( color )
    
    # get traced boundaries
    # boundaries = get_boundaries(kmeans_labels, unoverlap_labels_kmeans, width, height)
    

    
    
    
    return post_img, final_colors_Kmeans, add_mix_layers, mesh.vs # mesh.vs is palette



# visualize additive mixing
# input: per-pixel weights list
def get_additive_mixing_layers(add_mix_layers, width, height, palette, color):
    
    # select painting color first
    paint = palette[color]
    
    # initialize layer
    img_add_mix = np.zeros([width, height, 4])
    add_mix_layers = add_mix_layers.reshape(width, height, len(palette))
    for i in range(width):
        for j in range(height):
            img_add_mix[i, j, :3] = paint
            img_add_mix[i, j, 3] = add_mix_layers[i, j, color]
        
    return img_add_mix


def visualize_add_mix(add_mix_layers, img_w, img_h, palette, save_path):
    for i in range(palette.shape[0]):
        img_add_mix = get_additive_mixing_layers(add_mix_layers, img_w, img_h, palette, i)
        Image.fromarray(np.clip(0, 255, img_add_mix*255.).astype(np.uint8), 'RGBA').save(save_path + '-' + str(i) + '.png')



def main():
    import argparse
    parser = argparse.ArgumentParser( description = 'Posterization.' )
    parser.add_argument( 'input_image', help = 'The path to the input image.' )
    parser.add_argument( 'output_posterized_path', help = 'Where to save the output posterized image.' )
    parser.add_argument( 'output_add_mix_path', help = 'The path to the output additive-mixing image.' )
    #parser.add_argument( 'output_vectorized_path', help = 'The path to the output vectorized image.' )
    args = parser.parse_args()

    img_arr = np.asfarray( Image.open(args.input_image).convert( 'RGB' ) ) / 255.    #print(img_arr[1,1,:])
    img_arr_og = np.copy( img_arr )
    
    num_clusters = 20
    img_arr_re = img_arr.reshape( ( -1, 3 ) )
    img_arr_cluster = k_means_cluster( num_clusters, img_arr_re, img_arr.shape[0], img_arr.shape[1] )
    

    start = time.time()
    
    palette_num = 6
    num_blend = 3        # only 1 or 3 for now
    
    post_img, final_colors, add_mix_layers, palette = \
    posterization( args.input_image, img_arr_og, img_arr_cluster, palette_num, num_blend )
    
    Image.fromarray( np.clip( 0, 255, post_img*255. ).astype( np.uint8 ), 'RGB' ).save( args.output_posterized_path + '.jpg' )
 
    timg = simplepalettes.palette2swatch( palette )
    final_colors = np.array( final_colors )
    simplepalettes.save_image_to_file( np.clip( 0, 255, timg*255. ).astype( np.uint8 ), args.output_posterized_path + '-' + 'palette.png', clobber = True )
    simplepalettes.save_palette2wedges( final_colors, args.output_posterized_path + '-' + 'all_colors.png', clobber = True )
    
    visualize_add_mix( add_mix_layers, img_arr.shape[0], img_arr.shape[1], palette, args.output_add_mix_path )
    
    # width: 486, height: 864 for Kobe's example
    #Vectorized_image( args.output_vectorized_path, img_arr.shape[0], img_arr.shape[1], boundaries, final_colors )
    
    end = time.time()
    print( "Finished. Total time: ", end - start )
    
if __name__ == '__main__':
    main()

