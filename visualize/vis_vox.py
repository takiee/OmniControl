import trimesh
import numpy as np
import pyrender
import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# 加载mesh文件
mesh = trimesh.load('/root/code/seqs/object/001_book_1/simplified_scan_processed.obj')

# 将mesh体素化
# pitch 参数定义了每个体素的大小
voxels = mesh.voxelized(pitch=0.01)
print(voxels.shape)
print(voxels)
# 将体素化的mesh转换为体素矩阵
matrix = voxels.matrix

# 为了渲染，我们需要创建一个场景
scene = pyrender.Scene()

# 创建体素mesh的可视化对象
voxel_mesh = trimesh.voxel.ops.multibox(voxels.matrix, pitch=voxels.pitch)
voxel_mesh = pyrender.Mesh.from_trimesh(voxel_mesh)

# 将体素mesh添加到场景中
scene.add(voxel_mesh)

# 创建一个渲染器
renderer = pyrender.OffscreenRenderer(640, 480)

# 渲染场景
color, depth = renderer.render(scene)

# 显示渲染结果
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(color)
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(depth, cmap=plt.cm.gray_r)
plt.show()

# 如果需要，可以保存渲染的图片
# plt.savefig('rendered_mesh.png')
