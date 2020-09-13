import sys
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import coo_matrix
from scipy.optimize import *
from math import *
from trimesh import TriMesh
import cvxopt



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
#             volume=res['primal objective']+b.sum()
			
		
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
			mesh.vs = np.vstack( ( mesh.vs, final_point.clip(0.0,255.0) ) )
			

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
			print ('final vertices number', len(mesh.vs))
			break
	
	return mesh


def neighbor_cost_fun(s1, s2, l1, l2):
	
	return 0


"""
convert the 2d list of neighbors into two 1d numpy ndarrays s1, s2.
Each element in s1 should be smaller than the corresponding element in s2.
In our cases, number of ways of blendings will be 2.
"""
def neighbor_list_convert_to_ndarrays(neighbor_list, num_colors, num_of_ways):
	# compute total number of nodes in the graph
	num_nodes = num_colors + num_of_ways * (num_colors * (num_colors - 1)) / 2
	
	# may be slow, but easy to write for now
	neighbors_np = []
	for i in range(len(neighbor_list)):
		neighbor_list[i].sort()
		for j in range(len(neighbor_list[i])):
			if i < neighbor_list[i][j]:
				neighbors_np.append([i, neighbor_list[i][j]])
	
	neighbors_np = np.array(neighbors_np)
	s1 = neighbors_np[:, 0]
	s2 = neighbors_np[:, 1]
	#print(len(s1), len(s2))
	
	return s1, s2
	

def multi_label_opt(candidate_colors, neighbors_list, num_colors, num_of_ways):
	import gco
	gc = gco.GCO()
	
	#Create a general graph with specified number of sites and labels.
	gc.create_general_graph(len(candidate_colors), num_colors)
	
	"""
	Set unary potentials, unary should be a matrix of size
	nb_sites x nb_labels. unary can be either integers or float.
	"""
	gc.set_data_cost(np.array([[8, 1], [8, 2], [2, 8]]))
	
	# set up neighbor relationships
	# parameters: neighbor node 1, neighbor node 2, weights
	s1, s2 = neighbor_list_convert_to_ndarrays(neighbor_list, num_colors, num_of_ways)
	gc.set_all_neighbors(s1, s2, np.ones(len(s1)))
	
	
	
	return labels
	


def posterization(path, image_arr, num_colors):
	'''
	Given:
		image_arr: An n-rows by m-columns array of RGB colors.
		num_colors: number of ink colors for the poster.
	'''
	assert len(image_arr.shape) == 3
	assert num_colors == int(num_colors) and num_colors > 0
	
	
	'''
	#1 and #2
	'''
	
	# reshape image array into scipy.convexhull
	img_reshape = image_arr.reshape((-1,3))
	og_hull = ConvexHull(img_reshape)
	
	output_rawhull_obj_file = path + "-rawconvexhull.obj"
	write_convexhull_into_obj_file(og_hull, output_rawhull_obj_file)		
	
	# clipped already
	mesh = simplified_convex_hull(output_rawhull_obj_file, num_colors)
	
	# store the neighbors
	"""2-dimensional list"""
	neighbor_list = []
	for i in range(num_colors):
		neighbor_list.append(mesh.vertex_vertex_neighbors(i))
		
	
	# two-way discrete color blendings
	candidate_colors = mesh.vs
	pos_iter = num_colors
	for i in range(num_colors):
		for j in range(i+1, num_colors):
			# add to candidate colors
			candidate_colors = np.vstack((candidate_colors, .5*candidate_colors[i] + .5*candidate_colors[j]))
			candidate_colors = np.vstack((candidate_colors, .25*candidate_colors[i] + .75*candidate_colors[j]))
	
			# update neighbor list for the "first" blended color in original vertex's neighbor list
			neighbor_list[i].append(pos_iter)
			neighbor_list[j].append(pos_iter)	
			
			# update neighbor list for the "second" blended color in original vertex's neighbor list
			neighbor_list[i].append(pos_iter + 1)
			neighbor_list[j].append(pos_iter + 1)
			
			# add in neighbor list for our newly blended colors
			"""A lerp, so adjacent linear blended color will be a neighbor as well."""
			neighbor_list.append([i, j, pos_iter+1])
			neighbor_list.append([i, j, pos_iter])
			
			pos_iter += 2
	
	

def main():
	# (486, 864, 3) for kobe's example
	img_arr = np.asfarray(Image.open(sys.argv[1]).convert('RGB'))/255.
	posterization(sys.argv[1], img_arr, 6)
	#Image.fromarray(np.clip(0, 255, np.asfarray(arr)*255.)).save(sys.argv[2])

if __name__ == '__main__':
	main()
	