import sys
import numpy as np
from PIL import Image

from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import coo_matrix
from scipy.optimize import *
from skimage import color
from sklearn.cluster import KMeans

from math import *
from trimesh import TriMesh

import cvxopt
import potrace
import cairo
import random
import time

import simplepalettes
import extract_brush


# convexhull file I/O
def write_convexhull_into_obj_file(hull, output_rawhull_obj_file):
    hvertices=hull.points[hull.vertices]
    points_index=-1*np.ones(hull.points.shape[0],dtype=np.int32)
    points_index[hull.vertices]=np.arange(len(hull.vertices))
    #### start from index 1 in obj files!!!!!
    hfaces=np.array([points_index[hface] for hface in hull.simplices])+1
    
    #### to make sure each faces's points are countclockwise order.
    for index in range(len(hfaces)):
        face=hvertices[hfaces[index]-1]
        normals=hull.equations[index,:3]
        p0=face[0]
        p1=face[1]
        p2=face[2]
        
        n=np.cross(p1-p0,p2-p0)
        if np.dot(normals,n)<0:
            hfaces[index][[1,0]]=hfaces[index][[0,1]]
    
    myfile=open(output_rawhull_obj_file,'w')
    for index in range(hvertices.shape[0]):
        myfile.write('v '+str(hvertices[index][0])+' '+str(hvertices[index][1])+' '+str(hvertices[index][2])+'\n')
    for index in range(hfaces.shape[0]):
        myfile.write('f '+str(hfaces[index][0])+' '+str(hfaces[index][1])+' '+str(hfaces[index][2])+'\n')
    myfile.close()



def compute_tetrahedron_volume(face, point):
    n=np.cross(face[1]-face[0], face[2]-face[0])
    return abs(np.dot(n, point-face[0]))/6.0


# These function is directly copied from Jianchao Tan
def remove_one_edge_by_finding_smallest_adding_volume_with_test_conditions(mesh, option):
    edges=mesh.get_edges()
    faces=mesh.faces
    vertices=mesh.vs
    
    temp_list1=[]
    temp_list2=[]
    count=0

    for edge_index in range(len(edges)):
        
        edge=edges[edge_index]
        vertex1=edge[0]
        vertex2=edge[1]
        face_index1=mesh.vertex_face_neighbors(vertex1)
        face_index2=mesh.vertex_face_neighbors(vertex2)

        face_index=list(set(face_index1) | set(face_index2))
        related_faces=[faces[index] for index in face_index]
        old_face_list=[]
        
        
        #### now find a point, so that for each face in related_faces will create a positive volume tetrahedron using this point.
        ### minimize c*x. w.r.t. A*x<=b
        c=np.zeros(3)
        A=[]
        b=[]

        for index in range(len(related_faces)):
            face=related_faces[index]
            p0=vertices[face[0]]
            p1=vertices[face[1]]
            p2=vertices[face[2]]
            old_face_list.append(np.asarray([p0,p1,p2]))
            
            n=np.cross(p1-p0,p2-p0)
            
            #### Currently use this line. without this line, test_fourcolors results are not good.
            n=n/np.sqrt(np.dot(n,n)) ##### use normalized face normals means distance, not volume
            
            A.append(n)
            b.append(np.dot(n,p0))
            c+=n


        ########### now use cvxopt.solvers.lp solver
        A=-np.asfarray(A)
        b=-np.asfarray(b)
        
        c=np.asfarray(c)
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
        res = cvxopt.solvers.lp( cvxopt.matrix(c), cvxopt.matrix(A), cvxopt.matrix(b), solver='glpk' )

        if res['status']=='optimal':
            newpoint = np.asfarray( res['x'] ).squeeze()
        

            ######## using objective function to calculate (volume) or (distance to face) as priority.
            #volume=res['primal objective']+b.sum()
        
            ####### manually compute volume as priority,so no relation with objective function
            tetra_volume_list=[]
            for each_face in old_face_list:
                tetra_volume_list.append(compute_tetrahedron_volume(each_face,newpoint))
            volume=np.asarray(tetra_volume_list).sum()

            temp_list1.append((count, volume, vertex1, vertex2))
            temp_list2.append(newpoint)
            count+=1

    if option==1:
        if len(temp_list1)==0:
            print ('all fails')
            hull=ConvexHull(mesh.vs)
        else:
            min_tuple=min(temp_list1,key=lambda x: x[1])
            # print min_tuple
            final_index=min_tuple[0]
            final_point=temp_list2[final_index]
            # print 'final_point ', final_point
            new_total_points=mesh.vs
            new_total_points.append(final_point)

            hull=ConvexHull(np.array(new_total_points))
        return hull
    
    if option==2:
        
        if len(temp_list1)==0:
            # print 'all fails'
            pass
        else:
            min_tuple=min(temp_list1,key=lambda x: x[1])
            # print min_tuple
            final_index=min_tuple[0]
            final_point=temp_list2[final_index]
            # print 'final_point ', final_point
            
            v1_ind=min_tuple[2]
            v2_ind=min_tuple[3]

            ## Collect all faces touching the edge (either vertex).
            face_index1=mesh.vertex_face_neighbors(v1_ind)
            face_index2=mesh.vertex_face_neighbors(v2_ind)
            face_index=list(set(face_index1) | set(face_index2))
            ## Collect the vertices of all faces touching the edge.
            related_faces_vertex_ind=[faces[index] for index in face_index]

            ## Check link conditions. If link conditions are violated, the resulting
            ## mesh wouldn't be manifold.
            if len( (set(mesh.vertex_vertex_neighbors(v1_ind)).intersection(set(mesh.vertex_vertex_neighbors(v2_ind)))) ) != 2:
                print( "Link condition violated. Should not remove edge." )
            
            ## Remove the edge's two vertices.
            ## This also removes faces attached to either vertex.
            ## All other vertices have new indices.
            old2new=mesh.remove_vertex_indices([v1_ind, v2_ind])
            
            ## The edge will collapse to a new vertex.
            ## That new vertex will be at the end.
            new_vertex_index=current_vertices_num=len(old2new[old2new!=-1])
            
            ## Fill the hole in the mesh by re-attaching
            ## all the deleted faces to either removed vertex
            ## to the new vertex.
            new_faces_vertex_ind=[]
            
            for face in related_faces_vertex_ind:
                ## Map old vertex indices to new ones.
                ## The removed vertices both collapse to the new vertex index.
                new_face=[new_vertex_index if x==v1_ind or x==v2_ind else old2new[x] for x in face]
                ## The two faces on either side of the collapsed edge will be degenerate.
                ## Two vertices in those faces will both be the same vertex (the new one).
                ## Don't add that face.
                if len(set(new_face))==len(new_face):
                    new_faces_vertex_ind.append(new_face)
            
            
            ## Add the new vertex.
            ##### do not clip coordinates to[0,255]. when simplification done, clip.
            
            
            #### not sure if I clip this right, it may contain bugs
            mesh.vs = np.vstack( ( mesh.vs, final_point.clip(0.0,1.0) ) )
            

            ##### clip coordinates during simplification!
            #mesh.vs.append(final_point.clip(0.0,255.0))
            

            ## Add the new faces.
            # for face in new_faces_vertex_ind: mesh.faces.append(face)
            mesh.faces = np.vstack( ( mesh.faces, new_faces_vertex_ind ) )
            
            ## Tell the mesh to regenerate the half-edge data structure.
            mesh.topology_changed()
            # print (len(mesh.vs))
        return mesh

def simplified_convex_hull(output_rawhull_obj_file, num_colors):
    mesh = TriMesh.FromOBJ_FileName(output_rawhull_obj_file)
    print ('original vertices number:',len(mesh.vs))
    
    N = 500
    for i in range(N):
        #print ('loop:', i)
        old_num = len(mesh.vs)
        mesh = TriMesh.FromOBJ_FileName(output_rawhull_obj_file)
        mesh = remove_one_edge_by_finding_smallest_adding_volume_with_test_conditions(mesh,option=2)
        newhull = ConvexHull(mesh.vs)   # new convex hull
        
        write_convexhull_into_obj_file(newhull, output_rawhull_obj_file)
                
        if len(mesh.vs) == old_num or len(mesh.vs) <= num_colors:
            print ('final vertices number:', len(mesh.vs))
            break
    
    return mesh


def get_candidate_colors_and_neighbor_list(mesh, weight_list, num_colors, num_blend):
    a = 0.5
    b = 0.25
    c = 0.75
    blends = [[a, 1-a], [b, 1-b], [c, 1-c]]
    
    # store the neighbors
    """2-dimensional list"""
    neighbor_list = []
    for i in range(num_colors):
        neighbor_list.append(mesh.vertex_vertex_neighbors(i))
    
    #print(neighbor_list)
    
    # discrete blendings
    candidate_colors = mesh.vs
    pos_iter = num_colors
        
    for i in range(num_colors):
        for j in range(i+1, num_colors):
            if num_blend == 3:
                
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
                
            if num_blend == 1:
                    
                # add to candidate colors
                candidate_colors = np.vstack((candidate_colors, a*candidate_colors[i] + (1-a)*candidate_colors[j])) # middle blend
                
                neighbor_list[i].append(pos_iter)
                neighbor_list[j].append(pos_iter)
                
                neighbor_list.append([i, j])
                
                if i in neighbor_list[j]:
                    neighbor_list[i].remove(j)
                    neighbor_list[j].remove(i)
                
                weights = num_colors * [0]
                weights[i] = blends[0][0]
                weights[j] = blends[0][1]
                weight_list.append(weights)
                pos_iter += 1
        
    return candidate_colors, neighbor_list, np.array(weight_list)


# Unary term in minimizing energy function
def get_unary(image_arr, candidate_colors):
    
    # unary.shape = (486, 864, len(labels)) ideally!
    unary = np.zeros((image_arr.shape[0], image_arr.shape[1], len(candidate_colors)))
        
    for i in range(len(candidate_colors)):
        dist_square = (candidate_colors[i] - image_arr) ** 2
        unary[:, :, i] = np.sqrt(np.sum(dist_square, axis = 2))
        
        # LAB space dist:        
        #lab_cand = color.rgb2lab(candidate_colors[i])
        #lab_img = color.rgb2lab(image_arr)
        #dist_square = (lab_cand - lab_img) ** 2
        #unary[:, :, i] = 0.01 * np.sqrt(np.sum(dist_square, axis = 2)) 
        #print(np.amin(unary[:, :, i])) 
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
def share_same_palette(w1, w2):
    w1_nonzero = [i for i, e in enumerate(w1) if e != 0]
    w2_nonzero = [i for i, e in enumerate(w2) if e != 0]
    return w1_nonzero == w2_nonzero

def posterization(path, image_arr, num_colors, num_blend=1):
    '''
    Given:
        image_arr: An n-rows by m-columns array of RGB colors.
        num_colors: number of ink colors for the poster.
    '''
    assert len(image_arr.shape) == 3
    assert num_colors == int(num_colors) and num_colors > 0
    
    width = image_arr.shape[0]
    height = image_arr.shape[1]
    
    # reshape image array into scipy.convexhull
    img_reshape = image_arr.reshape((-1,3))
    og_hull = ConvexHull(img_reshape)
    output_rawhull_obj_file = path + "-rawconvexhull.obj"
    write_convexhull_into_obj_file(og_hull, output_rawhull_obj_file)		
    
    # get simplified convexhull
    # clipped already
    mesh = simplified_convex_hull(output_rawhull_obj_file, num_colors)

    # initialize weight list
    weight_list = []
    for i in range(num_colors):
        weight = num_colors * [0]
        weight[i] = 1
        weight_list.append(weight)
    
    
    # get blended colors and neighbor list
    num_blend=3
    candidate_colors, neighbor_list, weight_list_sep = \
    get_candidate_colors_and_neighbor_list(mesh, weight_list, num_colors, num_blend)
    
    
    # Multi-label optimization 
    import gco
    print('start multi-label optimization...')
    unary = get_unary(image_arr, candidate_colors)
    binary = get_binary(neighbor_list, weight_list_sep, candidate_colors, 0.1, 2)
        
    # get final labels from the optimization	
    labels = gco.cut_grid_graph_simple(unary, binary, n_iter=100, algorithm='swap') # alpha-beta-swap

    # locally optimize weights
    unoverlap_labels = get_unoverlap_labels(labels)
    weight_list_sep = optimize_weight(labels, unoverlap_labels, weight_list_sep, mesh.vs, image_arr)
    
    
    # compute final colors based on our optimized weights
    final_colors = []
    for label in unoverlap_labels:
        w = np.array(weight_list_sep[label])
        color = w @ mesh.vs
        final_colors.append(list(color))	
    '''
    
    final_colors = []
    for label in unoverlap_labels:
        final_colors.append(candidate_colors[label])
    '''
    print('Final colors being used:', len(final_colors))
    
    # convert labels to RGBs
    # visualize additive mixing layers
    post_img = np.zeros((labels.size, 3))
    add_mix_layers = np.zeros((labels.size, num_colors))
    for i in range(labels.size):
        post_img[i, :] = np.array(final_colors[unoverlap_labels.index(labels[i])])
        add_mix_layers[i, :] = np.array(weight_list_sep[unoverlap_labels.index(labels[i])])
    
    # reconstruct segmented images
    #post_img = np.asfarray(post_img).reshape(width, height, 3)
    
    
    #t = extract_brush.compute_stroke(image_arr, post_img)
    # K-means on RGBXY
    
    
    img_rgbxy = convertRGBXY(post_img, width, height)
    kmeans = KMeans(n_clusters=len(final_colors)-10, random_state=0).fit(img_rgbxy)
    kmeans_labels = kmeans.labels_
    clusters = kmeans.cluster_centers_
    post_img = np.zeros((labels.size, 3))
    for i in range(kmeans_labels.size):
        post_img[i, :] = clusters[kmeans_labels[i]][:3]
    post_img = np.asfarray(post_img).reshape(width, height, 3)
    
    
    return post_img, final_colors, add_mix_layers, mesh.vs # mesh.vs is palette



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
    args = parser.parse_args()

    img_arr = np.asfarray(Image.open(args.input_image).convert('RGB'))/255.    #print(img_arr[1,1,:])
    
    num_clusters = 20
    img_arr_re = img_arr.reshape((-1,3))
    img_arr_cluster = k_means_cluster(num_clusters, img_arr_re, img_arr.shape[0], img_arr.shape[1])
    

    start = time.time()
    
    palette_num = 6
    num_blend = 3        # only 1 or 3 for now
    
    post_img, final_colors, add_mix_layers, palette = \
    posterization(args.input_image, img_arr_cluster, palette_num, num_blend)
    
    Image.fromarray(np.clip(0, 255, post_img*255.).astype(np.uint8), 'RGB').save(args.output_posterized_path + '.jpg')
 
    timg = simplepalettes.palette2swatch(palette)
    final_colors = np.array(final_colors)
    simplepalettes.save_image_to_file(np.clip(0, 255, timg*255.).astype(np.uint8), args.output_posterized_path + '-' + 'palette.png', clobber=True)
    simplepalettes.save_palette2wedges(final_colors, args.output_posterized_path + '-' + 'all_colors.png', clobber=True)
    
    visualize_add_mix(add_mix_layers, img_arr.shape[0], img_arr.shape[1], palette, args.output_add_mix_path)
    
    end = time.time()
    print("Finished. Total time: ", end - start)
    
if __name__ == '__main__':
    main()

