import sys
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import coo_matrix
from scipy.optimize import *
from math import *
from trimesh import TriMesh
import cvxopt
import potrace
import cairo
import random

# Vectorzied a raster image
class Vectorized_image(object):
    def __init__(self, filename, width, height, boundaries, final_colors):
        self.surface = cairo.SVGSurface(filename + '.svg', height, width)
        cr = cairo.Context(self.surface)
        self.cr = cr
        
        # set up background
        cr.scale(height, width)
        cr.set_line_width(1)

        cr.rectangle(0, 0, 1, 1)
        cr.set_source_rgb(1, 1, 1) # white
        cr.fill()
        
        self.draw_dest_test(cr, height, width, boundaries, final_colors)

        self.surface.write_to_png(filename + '.png')
        cr.show_page()
        self.surface.finish()
    
        
    def draw_dest(self, cr, height, width, boundaries, final_colors):
        print("Start vectorizing images...")

        #print(len(final_colors))
        
        # Iterate over path curves
        for i in range(len(boundaries)):
            for curve in boundaries[i]:
                #print ("start_point =", curve.start_point)
                cr.move_to(curve.start_point[0]/height, curve.start_point[1]/width)
                for segment in curve:
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
                        
                #cr.line_to(curve.start_point[0]/height, curve.start_point[1]/width)
                cr.close_path()
                cr.set_source_rgba(final_colors[i][0], final_colors[i][1], final_colors[i][2], 1)
                cr.fill()
    
    def draw_dest_test(self, cr, height, width, boundaries, final_colors):
        print("Start vectorizing images...")

        #print(len(final_colors))
        
        # Iterate over path curves
        for i in range(len(boundaries)):
            painted = []
            for curve in boundaries[i]:
                #print ("start_point =", curve.start_point)
                cr.move_to(curve.start_point[0]/height, curve.start_point[1]/width)
                
                # draw parent curve
                if curve not in painted:
                    for segment in curve:
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

        #print ('current vertices number:', len(mesh.vs))
        
        if len(mesh.vs) == old_num or len(mesh.vs) <= num_colors:
            print ('final vertices number:', len(mesh.vs))
            break
    
    return mesh


def get_candidate_colors_and_neighbor_list(mesh, weight_list, num_colors):
    
    blends = [[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]]
    # store the neighbors
    """2-dimensional list"""
    neighbor_list = []
    for i in range(num_colors):
        neighbor_list.append(mesh.vertex_vertex_neighbors(i))
        
    # discrete blendings
    candidate_colors = mesh.vs
    pos_iter = num_colors
    for i in range(num_colors):
        for j in range(i+1, num_colors):
            # add to candidate colors
            candidate_colors = np.vstack((candidate_colors, .5*candidate_colors[i] + .5*candidate_colors[j]))
            candidate_colors = np.vstack((candidate_colors, .25*candidate_colors[i] + .75*candidate_colors[j]))
            candidate_colors = np.vstack((candidate_colors, .75*candidate_colors[i] + .25*candidate_colors[j]))
            
            # update neighbor list for the "first" blended color in original vertex's neighbor list
            #neighbor_list[i].append(pos_iter)
            #neighbor_list[j].append(pos_iter)
                
            # update neighbor list for the "second" blended color in original vertex's neighbor list
            neighbor_list[i].append(pos_iter + 1)
            #neighbor_list[j].append(pos_iter + 1)	
            
            # update neighbor list for the "third" blended color in original vertex's neighbor list
            #neighbor_list[i].append(pos_iter + 2)
            neighbor_list[j].append(pos_iter + 2)	
            
            # add in neighbor list for our newly blended colors
            """A lerp, so adjacent linear blended color will be a neighbor as well."""
            #neighbor_list.append([i, j])
            
            # add in neighbor list for our newly blended colors
            """A lerp, so adjacent linear blended color will be a neighbor as well."""
            neighbor_list.append([pos_iter+1, pos_iter+2])     # first color
            neighbor_list.append([i, pos_iter])
            neighbor_list.append([j, pos_iter])
            
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
    
    # unary.shape = (486, 864, 6) ideally!
    unary = np.zeros((image_arr.shape[0], image_arr.shape[1], len(candidate_colors)))
        
    for i in range(len(candidate_colors)):
        dist_square = (candidate_colors[i] - image_arr) ** 2
        unary[:, :, i] = np.sqrt(np.sum(dist_square, axis = 2))
        
    return unary
    

# Binary term in minimizing energy function
def get_binary(neighbor_list, weight_list, candidate_colors, option=1):
    
    # 1 if neighbor, (inf) if not neighbor, diag(binary) = 0
    if option == 1:
        binary = np.ones((len(neighbor_list), len(neighbor_list))) * np.inf
        
        for i in range(len(candidate_colors)):
            for j in range(len(candidate_colors)):
                if i == j:
                    binary[i][j] = 0
        
        for i in range(len(neighbor_list)):
            for j in range(len(neighbor_list[i])):
                binary[i][neighbor_list[i][j]] = 1
    
    # half of the L1 norm distance based on weights on each palette
    if option == 2:
        binary = np.zeros((len(neighbor_list), len(neighbor_list)))
        
        for i in range(len(candidate_colors)):
            for j in range(len(candidate_colors)):
                if i != j:
                    binary[i][j] = 0.5 * np.linalg.norm((weight_list[i] - weight_list[j]), ord=1)
                    
    return binary

# Get nonrepeated labels
def get_unoverlap_labels(colors):
    # option 1 means list
    visited = []
    for label in colors:
        if label not in visited:
            visited.append(label)
    return visited
    

# Trace the boundaries of different color regions
def get_boundaries(labels, img_h, img_w):
    
    boundaries = []
    
    # get unoverlapping label
    unoverlap_labels = get_unoverlap_labels(labels)
    num_label = len(unoverlap_labels)
    print('Final number of colors being used: ', num_label)
    
    # loop through all the labels and convert each label into a bitmap in each step
    for label in unoverlap_labels:
        spec_label = np.copy(labels)
        for i in range(spec_label.size):
            if spec_label[i] == label:   # if the pixel == label, we mark it
                spec_label[i] = 1
            else:						 # else: we do not trace it!
                spec_label[i] = 0
        spec_label_pos = spec_label.reshape(img_h, img_w)
        
        # Create a bitmap from the array
        bmp = potrace.Bitmap(spec_label_pos)

        # Trace the bitmap
        path = bmp.trace()
        boundaries.append(path)
        
    return boundaries, unoverlap_labels


# visualize additive mixing
# input: per-pixel weights list
def get_additive_mixing_layers(add_mix_layers, width, height, palette, color):
    
    # select painting color first
    paint = palette[color]
    
    # initialize layer
    img_add_mix = np.zeros([width, height, 4])
    add_mix_layers = add_mix_layers.reshape(width, height, 6)
    for i in range(width):
        for j in range(height):
            img_add_mix[i, j, :3] = paint
            img_add_mix[i, j, 3] = add_mix_layers[i, j, color]
        
    return img_add_mix




def posterization(path, image_arr, num_colors):
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
    
    # clipped already
    mesh = simplified_convex_hull(output_rawhull_obj_file, num_colors)

    # compute weight list
    weight_list = []
    for i in range(num_colors):
        weight = num_colors * [0]
        weight[i] = 1
        weight_list.append(weight)
    
    # get blended colors and neighbor list
    candidate_colors, neighbor_list, weight_list = \
    get_candidate_colors_and_neighbor_list(mesh, weight_list, num_colors)
    
    #print(weight_list)
    
    # Multi-label optimization 
    import gco
    print('start multi-label optimization...')
    unary = get_unary(image_arr, candidate_colors)
    binary = get_binary(neighbor_list, weight_list, candidate_colors, option=1)
    
    # get final labels from the optimization	
    labels = gco.cut_grid_graph_simple(unary, binary, n_iter=100, algorithm='swap') # alpha-beta-swap
    
    
    # convert labels to RGBs
    # visualize additive mixing layers
    
    post_img = np.zeros((labels.size, 3))
    add_mix_layers = np.zeros((labels.size, num_colors))
    for i in range(labels.size):
        post_img[i, :] = np.array(candidate_colors[labels[i]])
        add_mix_layers[i, :] = np.array(weight_list[labels[i]]) 

    
    # reconstruct segmented images
    post_img = np.asfarray(post_img).reshape(width, height, 3)
    
    ########
    ########
    boundaries, unoverlap_labels = get_boundaries(labels, width, height)
    
    final_colors = []
    for label in unoverlap_labels:
        final_colors.append(candidate_colors[label])
    
        
    return post_img, boundaries, final_colors, add_mix_layers, mesh.vs # mesh.vs is palette
    
    
    

def main():
    import argparse
    parser = argparse.ArgumentParser( description = 'Posterization.' )
    parser.add_argument( 'input_image', help = 'The path to the input image.' )
    parser.add_argument( 'output_posterized_path', help = 'Where to save the output posterized image.' )
    parser.add_argument( 'output_vectorized_path', help = 'The path to the output vectorized image.' )
    #parser.add_argument( 'output_add_mix_path', help = 'The path to the output additive-mixing image.' )
    args = parser.parse_args()

    img_arr = np.asfarray(Image.open(args.input_image).convert('RGB'))/255.    #print(img_arr[1,1,:])
    post_img, boundaries, final_colors, add_mix_layers, palette = posterization(args.input_image, img_arr, 6)
    
    Image.fromarray(np.clip(0, 255, post_img*255.).astype(np.uint8), 'RGB').save(sys.argv[2])
    
    # width: 486, height: 864 for Kobe's example
    Vectorized_image(args.output_vectorized_path, img_arr.shape[0], img_arr.shape[1], boundaries, final_colors)
    '''
    # visualize additive mixing layer
    img_add_mix = get_additive_mixing_layers(add_mix_layers, img_arr.shape[0], img_arr.shape[1], palette, color=5)
    Image.fromarray(np.clip(0, 255, img_add_mix*255.).astype(np.uint8), 'RGBA').save(sys.argv[4])
    '''
if __name__ == '__main__':
    main()



'''
# helper functions for annulus_sort
# check if the sub_curve is inside the curve
def check_annulus(self, sub_curve, curve):
    
    x, y = sub_curve.start_point[0], sub_curve.start_point[1]
    
    control_x = []
    control_y = []
    for segment in curve:
        end_point_x, end_point_y = segment.end_point
        control_x.append(end_point_x)
        control_y.append(end_point_y)
    
    if min(control_x) <= x <= max(control_x) and min(control_y) <= y <= max(control_y):
        return True
   
    return False
    
# Checking if there is a sub-curve inside the curve
def annulus_sort(self, curve_and_colors):
    
    # buuble sort based on annulus-check
    n = len(curve_and_colors)
    for i in range(n-1):
        for j in range(0, n-1-i):
            if self.check_annulus(curve_and_colors[j][0], curve_and_colors[j+1][0]):
                curve_and_colors[j], curve_and_colors[j+1] = \
                 curve_and_colors[j+1], curve_and_colors[j] 
                
    return curve_and_colors

# For getting the coloring orders correct
def reorder_coloring(self, boundaries, final_colors):
    curve_and_colors = []
    for i in range(len(boundaries)):
        for curve in boundaries[i]:
            curve_and_colors.append([curve, final_colors[i]])
    
    return self.annulus_sort(curve_and_colors)
    
    
def vectorize_regions(self, cr, height, width, ordered_curve_and_colors):
    print("Start vectorizing images...")
    
    for region in ordered_curve_and_colors:
        curve = region[0]
        color = region[1]
        print ("start_point =", curve.start_point)
        cr.move_to(curve.start_point[0]/height, curve.start_point[1]/width)
        for segment in curve:
            print (segment)
            end_point_x, end_point_y = segment.end_point
            
            if segment.is_corner:
                c_x, c_y = segment.c
                cr.line_to(c_x/height, c_y/width)
            else:
                c1_x, c1_y = segment.c1
                c2_x, c2_y = segment.c2
                cr.curve_to(c1_x/height, c1_y/width, c2_x/height, c2_y/width, \
                end_point_x/height, end_point_y/width)
                
        #cr.line_to(curve.start_point[0]/height, curve.start_point[1]/width)
        cr.close_path()
        cr.set_source_rgba(color[0], color[1], color[2], 1)
        cr.fill()
            
def has_children(self, boundaries, final_colors):
    parent_bd = []
    for i in range(len(boundaries)):
        boundaries[i] = [boundaries[i], final_colors[i]]
        for region in boundaries[i]:
            curve = region[0]
            color = region[1]
            if curve.children:
                parent_bd.append(boundaries.pop(i))
            
'''