import cv2
import numpy as np
import open3d as o3d
import os
from utils import bbox3d2corners

COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
COLORS_IMG = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]

LINES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [2, 6],
    [7, 3],
    [1, 5],
    [4, 0]
]


def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    density = npy[:, 3]
    colors = [[item, item, item] for item in density]
    ply.colors = o3d.utility.Vector3dVector(colors)
    return ply


def ply2npy(ply):
    return np.array(ply.points)


def bbox_obj(points, color=[1, 0, 0]):
    colors = [color for i in range(len(LINES))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def vis_core(plys):
    o3d.visualization.draw_plotly(plys, window_name='Open3D', width=600, height=400,
                                  mesh_show_wireframe=False, point_sample_factor=1, front=None, lookat=None, up=None,
                                  zoom=1.0)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    #
    # PAR = os.path.dirname(os.path.abspath(__file__))
    # ctr = vis.get_view_control()
    #
    # param = o3d.io.read_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'))
    #
    # for ply in plys:
    #     vis.add_geometry(ply)
    #
    # ctr.convert_from_pinhole_camera_parameters(param)
    #
    # vis.run()
    # # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # # o3d.io.write_pinhole_camera_o3d.visualizationparameters(os.path.join(PAR, 'viewpoint.json'), param)
    # vis.destroy_window()


def vis_pc(pc, bboxes=None, labels=None):
    '''
    pc: ply or np.ndarray (N, 4)
    bboxes: np.ndarray, (n, 7) or (n, 8, 3)
    labels: (n, )
    '''
    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0])

    if bboxes is None:
        vis_core([pc, mesh_frame])
        return

    if len(bboxes.shape) == 2:
        bboxes = bbox3d2corners(bboxes)

    vis_objs = [pc, mesh_frame]
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        if labels is None:
            color = [1, 1, 0]
        else:
            if labels[i] >= 0 and labels[i] < 3:
                color = COLORS[labels[i]]
            else:
                color = COLORS[-1]
        vis_objs.append(bbox_obj(bbox, color=color))
    vis_core(vis_objs)


def vis_img_3d(img, image_points, labels, rt=True):
    '''
    img: (h, w, 3)
    image_points: (n, 8, 2)
    labels: (n, )
    '''

    for i in range(len(image_points)):
        label = labels[i]
        bbox_points = image_points[i]  # (8, 2)
        if label >= 0 and label < 3:
            color = COLORS_IMG[label]
        else:
            color = COLORS_IMG[-1]
        for line_id in LINES:
            x1, y1 = bbox_points[line_id[0]]
            x2, y2 = bbox_points[line_id[1]]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
    if rt:
        return img
    cv2.imshow('bbox', img)
    cv2.waitKey(0)


def vis_kitti_pts(point_cloud_np, file_name):
    try:
        # 提取空间坐标和强度值
        xyz = point_cloud_np[:, :3]
        intensity = point_cloud_np[:, 3]
        #
        # # 标准化强度值到0-1范围
        intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        #
        # # # 创建颜色数组，假设强度值映射到灰度颜色
        # colors = np.full(xyz.shape[0], intensity_normalized)
        # colors = colors.astype(np.float32)  # 确保颜色数组的数据类型是float32

        # 创建Open3D的点云对象，并设置点的位置和颜色
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector4iVector(point_cloud_np)
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        # 可视化点云
        print("pcd: ", pcd)
        o3d.io.write_point_cloud(f"{file_name}.pcd", pcd)
        print(f"{file_name}.pcd")
    except Exception as e:
        print(e)
