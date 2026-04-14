
import numpy as np
import pandas as pd


def custom_ravel_multi_index(multi_index, dims):
    """
    Custom implementation of np.ravel_multi_index.

    Parameters:
    - multi_index: numpy array of shape (N, dim), where N is the number of points
                   and dim is the number of dimensions. Each row contains the
                   multi-dimensional index of a point.
    - dims: tuple or array_like representing the dimensions of the grid.

    Returns:
    - h: numpy array of shape (N,) containing the flattened indices for each point.
    """
    # Validate input shapes

    if multi_index.shape[1] != len(dims):
        raise ValueError("Shape of multi_index must match length of dims tuple")

    # Compute the flattened indices
    h = np.zeros(multi_index.shape[0], dtype=np.intp)
    for i in range(len(dims)):
        h *= dims[i]
        h += multi_index[:, i]

    return h

def closest_to_center(data):
    # 转换成NumPy数组以便于处理
    data_np = np.array(data)

    # 计算每个维度的平均值
    mean_values = np.mean(data_np, axis=0)

    # 排序数据，选择每一维度上与平均值最接近的数据点
    sorted_data = sorted(data, key=lambda x: sum((xi - mi) ** 2 for xi, mi in zip(x, mean_values)))

    # 返回最靠近平均值的数据点（如果有多个，返回第一个）
    return sorted_data[0]

def voxel_filter(point_cloud, leaf_size=0.05, random=False):

    if leaf_size != 0:
        num_dims = point_cloud.shape[1]  # Number of dimensions in the point cloud (assuming it's 32 for your case)

        # Calculate minimum and maximum values across all dimensions
        # mins = np.amin(point_cloud, axis=0)
        maxs = np.amax(point_cloud, axis=0)

        mins = np.zeros(point_cloud.shape[1])
        # mins=np.amin(point_cloud,axis=0)
        maxs = np.ceil(maxs)
        # mins = np.ceil(mins)

        # Calculate voxel grid dimensions
        D = ((maxs - mins) // leaf_size + 1).astype(int)
        # D_prod = np.prod(D)  # Total number of voxels

        # Calculate voxel indices for each point
        voxel_indices = np.zeros((point_cloud.shape[0], num_dims), dtype=np.int32)
        for i in range(num_dims):
            shifted = point_cloud[:, i] - mins[i]
            indices = (shifted // leaf_size).astype(int)
            voxel_indices[:, i] = indices

        # Combine voxel indices into a single index per point (assuming a linearized index)
        h = custom_ravel_multi_index(voxel_indices, D)

        # Filter points based on voxel indices
        filtered_points = []
        h_indice = np.argsort(h)
        begin = 0
        for i in range(len(h_indice) - 1):
            if h[h_indice[i]] == h[h_indice[i + 1]]:
                continue
            else:
                point_idx = h_indice[begin:i + 1]
                filtered_points.append(closest_to_center(point_cloud[point_idx]))
                begin = i + 1

        # Handle the last group of points
        point_idx = h_indice[begin:]
        filtered_points.append(closest_to_center(point_cloud[point_idx]))

        # Convert filtered points to NumPy array
        filtered_points = np.array(filtered_points, dtype=np.float64)
        column_names = ['Ag', 'Cu', 'Zn', 'Cd', 'In', 'Sn', 'Ga', 'Bi', 'Ni', 'Ti', 'Sb', 'Dy', 'Mn', 'Al', 'Y', 'Ce',
                        'Co']

        filtered_points = pd.DataFrame(filtered_points, columns=column_names)

        point_out = filtered_points


    else:
        point_out = point_cloud
        # Convert filtered points to NumPy array
    filtered_points = np.array(point_out, dtype=np.float64)
    column_names = ['Ag', 'Cu', 'Zn', 'Cd', 'In', 'Sn', 'Ga', 'Bi', 'Ni', 'Ti', 'Sb', 'Dy', 'Mn', 'Al', 'Y', 'Ce',
                        'Co']
    filtered_points = pd.DataFrame(filtered_points, columns=column_names)

    return filtered_points