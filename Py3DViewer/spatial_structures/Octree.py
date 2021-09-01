import numpy as np
from numba import njit, objmode, boolean, int64
from numba.experimental import jitclass
from numba.types import ListType as LT
from numba.typed import List as L
import numba
from ..geometry import AABB, SpaceObject

spec_noctree_node = [
    ('father', int64),
    ('depth', int64),
    ('items', int64[:]),
    ('children', int64[:]),
]

@jitclass(spec_noctree_node)
class NOctreeNode:
    
    def __init__(self,father, depth, items):
        self.father   = father
        self.depth    = depth
        self.items    = items
        self.children = np.zeros(8, dtype=np.int64)-1
        
    
    @property
    def is_leaf(self):
        return self.children[0] == -1


node_type = NOctreeNode.class_type.instance_type
a_type = AABB.class_type.instance_type
so_type = SpaceObject.class_type.instance_type


spec_noctree = [
    ('__items', LT(so_type)),
    ('__max_depth', int64),
    ('__items_per_leaf', int64),
    ('__nodes', LT(node_type)),
    ('__aabbs', LT(a_type)),
    ('__depth', int64)
    
]

@jitclass(spec_noctree)
class NOctree:
    
    def __init__(self, items, max_depth=8, items_per_leaf=10):
        
        self.__max_depth      = max_depth
        self.__items_per_leaf = items_per_leaf
        self.__nodes = numba.typed.List.empty_list(node_type)
        self.__aabbs = numba.typed.List.empty_list(a_type)
        self.__items = items
        self.__depth = 0
        
        
    
    @property
    def nodes(self):
        return self.__nodes
    
    @property
    def aabbs(self):
        return self.__aabbs
    
    @property
    def max_depth(self):
        return self.__max_depth
    
    @property
    def items_per_leaf(self):
        return self.__items_per_leaf
    
    @property
    def depth(self):
        return self.__depth
              
@njit(int64(NOctree.class_type.instance_type, int64), cache=True)
def create_new_aabbs(t, idx):
    min_ = t._NOctree__aabbs[idx].min
    max_ = t._NOctree__aabbs[idx].max
    center = t._NOctree__aabbs[idx].center
        
        
    c1 = AABB(np.array([[min_[0], min_[1], min_[2]], [center[0], center[1], center[2]]], dtype=np.float64))
    c2 = AABB(np.array([[center[0], min_[1], min_[2]], [max_[0], center[1], center[2]]], dtype=np.float64))
    c3 = AABB(np.array([[center[0], center[1], min_[2]], [max_[0], max_[1], center[2]]], dtype=np.float64))
    c4 = AABB(np.array([[min_[0], center[1], min_[2]], [center[0], max_[1], center[2]]], dtype=np.float64))
    c5 = AABB(np.array([[min_[0], min_[1], center[2]], [center[0], center[1], max_[2]]], dtype=np.float64))
    c6 = AABB(np.array([[center[0], min_[1], center[2]], [max_[0], center[1], max_[2]]], dtype=np.float64))
    c7 = AABB(np.array([[center[0], center[1], center[2]], [max_[0], max_[1], max_[2]]], dtype=np.float64))
    c8 = AABB(np.array([[min_[0], center[1], center[2]], [center[0], max_[1], max_[2]]], dtype=np.float64))

    for c in [c1,c2,c3,c4,c5,c6,c7,c8]:
        t._NOctree__aabbs.append(c)
            
    return 0

@njit(int64(NOctree.class_type.instance_type, int64), cache=True)
def build_octree(t, verts_per_item):
    

    vertices = np.zeros((len(t._NOctree__items)*verts_per_item, 3), dtype=np.float64)
    for  idx, el in enumerate(t._NOctree__items):
        v = el.vertices
        for i in range(verts_per_item):
            vertices[idx*verts_per_item+i] = v[i]
    
    root_aabb = AABB(vertices)
    

    its = np.array(list(range(0,len(t._NOctree__items))), dtype=np.int64)
    root_node = NOctreeNode(-1, 0, its)
    t._NOctree__nodes.append(root_node)
    t._NOctree__aabbs.append(root_aabb)
    
    aabbs = [AABB(t._NOctree__items[idx].vertices) for idx in its]
        
    queue = L()
    queue.append(0)
    while len(queue) > 0:
        curr = queue.pop(0)
        
        items = t._NOctree__nodes[curr].items
                
        if items.shape[0] > t._NOctree__items_per_leaf and t._NOctree__nodes[curr].depth < t._NOctree__max_depth:
                #split node
                create_new_aabbs(t, curr)
                for aabb in t._NOctree__aabbs[-8:]:
                    items_in_node = []
                    for idx in items:
                        if aabb.intersects_box(aabbs[idx]):
                            items_in_node.append(idx)
                    new_node = NOctreeNode(curr, t._NOctree__nodes[curr].depth+1, np.array(items_in_node, dtype=np.int64))
                    t._NOctree__nodes.append(new_node)
                
                t._NOctree__nodes[curr].items = np.empty(0, dtype=np.int64) #free memory
                if t._NOctree__nodes[curr].depth+1 > t._NOctree__depth:
                    t._NOctree__depth = t._NOctree__nodes[curr].depth+1
                
                for idx, n in enumerate(range(len(t._NOctree__nodes)-8,  len(t._NOctree__nodes))):
                    queue.append(n)
                    t._NOctree__nodes[curr].children[idx] = n
    
    return 0

@njit(cache=True)            
def search_p(nodes,shapes,aabbs,point,type_mesh,index=0):
        
        aabb = aabbs[index]
        
        if (point[0] < aabb.min[0]
            or point[0] > aabb.max[0]
            or point[1] < aabb.min[1]
            or point[1] > aabb.max[1]
            or point[2] < aabb.min[2]
            or point[2] > aabb.max[2]):
                return False,-1
        
        if point[0] <= aabb.center[0]:
            if point[1] <= aabb.center[1]:
                if point[2] <= aabb.center[2]:
                    inner_child_index = 0 #left_back_bottom
                else:
                    inner_child_index = 4 #left_front_bottom
            
            else:
                if point[2] <= aabb.center[2]:
                    inner_child_index = 3 #left_back_top
                else:
                    inner_child_index = 7 #left_front_top
        else:
            if point[1] <= aabb.center[1]:
                if point[2] <= aabb.center[2]:
                    inner_child_index = 1 #right_back_bottom
                else:
                    inner_child_index = 5 #right_front_bottom
            else:
                if point[2] <= aabb.center[2]:
                    inner_child_index = 2 #right_back_top
                else:
                    inner_child_index = 6 #right_front_top
                    
        nodes_child_index = nodes[index].children[inner_child_index]
        if len(nodes[nodes_child_index].items) == 0 and not np.array_equal(nodes[nodes_child_index].children, np.zeros(8)):
            return search_p(nodes, shapes, aabbs, point,type_mesh,nodes_child_index)
        else:
            if 'Trimesh' in type_mesh:
                for i in nodes[nodes_child_index].items:
                    if shapes[i].triangle_contains_point(point):
                        return True,i
            
            elif 'Quadmesh' in type_mesh:
                for i in nodes[nodes_child_index].items:
                    if shapes[i].quad_contains_point(point):
                        return True,i
                    
            elif 'Hexmesh' in type_mesh:
                for i in nodes[nodes_child_index].items:
                    if shapes[i].hex_contains_point(point):
                        return True,i
                    
            elif 'Tetmesh' in type_mesh:
                for i in nodes[nodes_child_index].items:
                    if shapes[i].tet_contains_point(point):
                        return True,i
                
            return False,-1

#Method to get the items intersected by a ray.
#We check if the ray intersects the AABB and if true we add the node to the queue of nodes and check which octants it #intersects. If the node intersected is a leave we add it to the queue. If the node is not a leave we add every item #intersected by the ray to a set
@njit(cache=True)
def intersects_ray(nodes, shapes, aabbs, r_origin, r_dir, type_mesh):
    shapes_hit = set()
    
    if(not aabbs[0].intersects_ray(r_origin, r_dir)):
        return False, shapes_hit
    
    q = List()
    q.append(0)

    while(len(q)>0):
        index = q.pop(0)
        node = nodes[index]
        
        if not(node.children == np.zeros(8)).all():
            for c in node.children:
                child = nodes[c]
                if(aabbs[c].intersects_ray(r_origin, r_dir)):
                    if(len(child.items) == 0):
                        q.append(c)
                    else:
                        if 'Trimesh' in type_mesh:
                            for i in child.items:
                                if(shapes[i].ray_interesects_triangle(r_origin, r_dir)):
                                    shapes_hit.add(i)
                        elif 'Quadmesh' in type_mesh:
                            for i in child.items:
                                if(shapes[i].ray_interesects_quad(r_origin, r_dir)):
                                    shapes_hit.add(i)
                        elif 'Hexmesh' in type_mesh:
                            for i in child.items:
                                if(shapes[i].ray_interesects_hex(r_origin, r_dir)):
                                    shapes_hit.add(i)
                        elif 'Tetmesh' in type_mesh:
                            for i in child.items:
                                if(shapes[i].ray_interesects_tet(r_origin, r_dir)):
                                    shapes_hit.add(i)
    if(len(shapes_hit) == 0):
        return False,shapes_hit
    return True,shapes_hit

@njit
def build_space_objects(polys):
    l = L()
    for p in polys:
        l.append(SpaceObject(p))
    
    return l

class Octree:
    def __init__(self,mesh=None, items_per_leaf=10, max_depth=8):
        
        if mesh is not None:
            self.mesh_type = str(type(mesh))
            polys = mesh.vertices[mesh.polys]
            self.items = build_space_objects(polys)       
        else:
            print("Error")
         
        verts_per_item = self.items[0].vertices.shape[0]
        self.__tree = NOctree(self.items, max_depth, items_per_leaf)
        build_octree(self.__tree, verts_per_item)

    def search_point(self,point):
        return search_p(self.__tree.nodes,self.items,self.__tree._NOctree__aabbs,point,self.mesh_type,index=0)
    
    def intersects_ray(self, r_origin, r_dir):
        return intersects_ray(self.__tree.nodes, self.items, self.__tree._NOctree__aabbs, r_origin, r_dir, self.mesh_type)
    
    @property
    def nodes(self):
        return self.__tree._NOctree__nodes
    
    @property
    def aabbs(self):
        return self.__tree._NOctree__aabbs
    
    @property
    def max_depth(self):
        return self.__tree._NOctree__max_depth
    
    @property
    def items_per_leaf(self):
        return self.__tree._NOctree__items_per_leaf
    
    @property
    def depth(self):
        return self.__tree._NOctree__depth