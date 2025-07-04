# Based on a script provided to the FEFLOW community by Dr Carlos Andres Rivera Villarreyes

import numpy as np
import pandas as pd
import geopandas as gpd
import pyvista as pv

file_name = "Topography"
input_path = r"c:\Users\Anton\OneDrive\Code\PythonForHydrogeologists\data\GIS\topography.csv"

pdTopo = pd.read_csv(input_path, delimiter=';', header=0)
x_array = np.array(pdTopo.X)
y_array = np.array(pdTopo.Y)
z_array = np.array(pdTopo.Z)

"""
# Workflow for shapefiles
# For shapefiles use GeoPandas
df_input_file = gpd.read_file(input_path)

# Create emtpy lists to collect point information
x_list = []
y_list = []
z_list = []

# Iterate over the points
i = 0
for index, _ in df_input_file.iterrows():
    # Collecting the xyz information
    x, y, z = df_input_file.loc[index].geometry.x, df_input_file.loc[index].geometry.y, df_input_file.loc[index].geometry.z
    # And storing them in lists
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)
    i+=1

# Convert lists to numpy arrays
x_array = np.array(x_list)
y_array = np.array(y_list)
z_array = np.array(z_list)
"""

# Convert to points data
points = np.column_stack((x_array, y_array, z_array))
# Convert the point data to a point cloud
point_cloud = pv.PolyData(points)
# Mesh the points as a 2D Delauny triangulated surface
mesh = point_cloud.delaunay_2d(tol=1E-3)
# Smooth the mesh
mesh = mesh.smooth(n_iter=1000, relaxation_factor=0.01, edge_angle=15, feature_angle=15, boundary_smoothing=True)
# Save the mesh
mesh.save(f".\output\{file_name}.stl")
# Display the mesh
pl = pv.Plotter()
_ = pl.add_mesh(mesh)
pl.show()
