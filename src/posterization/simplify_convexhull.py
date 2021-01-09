import numpy as np
import cvxopt

from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import coo_matrix
from scipy.optimize import *
from .trimesh import TriMesh


# convexhull file I/O
def get_faces_vertices(hull):
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
    
    #print('v: ', hvertices[0])
    #print('h: ', hfaces)
    return hvertices, hfaces
    
    '''
    myfile=open(output_rawhull_obj_file,'w')
    for index in range(hvertices.shape[0]):
        myfile.write('v '+str(hvertices[index][0])+' '+str(hvertices[index][1])+' '+str(hvertices[index][2])+'\n')
    for index in range(hfaces.shape[0]):
        myfile.write('f '+str(hfaces[index][0])+' '+str(hfaces[index][1])+' '+str(hfaces[index][2])+'\n')
    myfile.close()
    '''

# Turn hvertices, hfaces into a mesh object


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

### ideally: def simplified_convex_hull(num_colors, hvertices, hfaces)
def simplified_convex_hull(num_colors, vertices, faces):
    mesh = TriMesh.FromVertexFace_to_MeshObj( vertices, faces )
    print ('original vertices number:',len(mesh.vs))
    
    if num_colors < len( mesh.vs ):
        N = 500
        for i in range(N):
            #print ('loop:', i)
            old_num = len(mesh.vs)
            mesh = remove_one_edge_by_finding_smallest_adding_volume_with_test_conditions(mesh,option=2)
            newhull = ConvexHull(mesh.vs)   # new convex hull
            
                    
            if len(mesh.vs) == old_num or len(mesh.vs) <= num_colors:
                break
    print ('final vertices number:', len(mesh.vs))
    return mesh
