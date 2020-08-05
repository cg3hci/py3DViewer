def mesh_is_volumetric(mesh):
    return hasattr(mesh, 'tets') or hasattr(mesh, 'hexes')

def mesh_is_surface(mesh):
    return not mesh_is_volumetric(mesh)