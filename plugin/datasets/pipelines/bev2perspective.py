import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from typing import List, Tuple, Union, Dict
import cv2
import re
import pdb
import hashlib
import json
import copy


@PIPELINES.register_module(force=True)
class ExtractBEV2Perspective(object):
    """Generate FOV base line and store it under
    'cam2bev_fov_indices' key.

    Args:
        roi_size (tuple or list): bev range (this tells us about how to resize or adjust the seg or line output, which are ego centric global coordinates)
        bev_array_size (tuple or list): bev feature size
        img_size (tuple or list): height, width (for intrinsics related calculations)
        return_same (boolean): not all frames have the same intrinsics and extrinsics, so ...
    """

    def __init__(self,
                 roi_size: Union[Tuple, List],
                 bev_array_size: Union[Tuple, List],
                 perspective_grids: Union[Tuple, List],
                 height_levels: Dict[str, int],
                 allow_spatial_overlap: bool,
                 dataset="nuscenes",
                 precision=-1, # in meters, like 0.15
                 ):
        self.precision = precision
        self.historical_results = dict()
        self.not_found_count = 0
        self.roi_size = roi_size
        self.bev_array_size = bev_array_size
        self.bev_h = bev_array_size[0]
        self.bev_w = bev_array_size[1]
        self.perspective_grid_x, self.perspective_grid_y = perspective_grids
        self.height_levels = height_levels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.allow_spatial_overlap = allow_spatial_overlap

    def extract_forward_vector(self, const_ego2cam):
        # input is extrinsics from dataloader
        # returns:
        #   zero_centered forward direction,
        #   zero_centered up direction,
        #   and camera_position:
        #   (a tuple of 3 numpy array) or (a list of tuples of 3 numpy array)
        # the reference point is ego, so the coordinates returned are wrt to ego

        # the units to be processes are -> 4*4
        if not isinstance(const_ego2cam, np.ndarray):
            ego2cam = np.array(const_ego2cam)
        else:
            ego2cam = const_ego2cam.copy()

        dims = len(ego2cam.shape)
        if dims == 2:
            cam2ego = np.linalg.inv(ego2cam)

            transformed_origin = (cam2ego @ np.array([0, 0, 0, 1]))[0:3]
            transformed_forward = (cam2ego @ np.array([0, 0, 1, 1]))[0:3]
            transformed_up = (cam2ego @ np.array([0, -1, 0, 1]))[0:3]

            return transformed_forward - transformed_origin, transformed_up - transformed_origin, transformed_origin

        assert dims == 3
        results = list()
        for matrix in ego2cam:
            cam2ego = np.linalg.inv(matrix)

            transformed_origin = (cam2ego @ np.array([0, 0, 0, 1]))[0:3]
            transformed_forward = (cam2ego @ np.array([0, 0, 1, 1]))[0:3]
            transformed_up = (cam2ego @ np.array([0, -1, 0, 1]))[0:3]

            results.append(
                (transformed_forward - transformed_origin, transformed_up - transformed_origin, transformed_origin))

        return results

    def calculate_image_plane_corners(self, K, forward, up, camera_pos, width, height):
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]

        # Distance to the image plane (using normalized coordinates)
        d = 1

        # Define the corners in the local camera frame
        # it isn't important whether you use resized intrinsics or not, bc the fx and cx are multiplied with the same
        # number during resizing (when change intrinsics flag is set), and the division cx/fx yields in the same value
        # same applies to fy and cy
        # the extrinsics doesn't change anyways
        corners_local = np.array([
            np.array([-(cx / fx), d, (cy / fy), 1]),  # top-left
            np.array([(width - cx) / fx, d, (cy / fy), 1]),  # top-right
            np.array([(width - cx) / fx, d, -(height - cy) / fy, 1]),  # bottom-right
            np.array([-(cx / fx), d, -(height - cy) / fy, 1])  # bottom-left
        ])

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        forward = forward / np.linalg.norm(forward)
        up = up / np.linalg.norm(up)
        translation = camera_pos

        base2accurate = np.array([right, forward, up, translation])
        corners_world = corners_local @ base2accurate

        return corners_world
    
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2 using homogeneous coordinates """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        # Create the rotation matrix 3x3
        R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        # Convert to 4x4 homogeneous matrix
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = R

        return rotation_matrix

    def intersection_with_xy_plane(self, start, end):
        """
        Find the intersection point of a vector from start to end with the XY plane (z = 0).

        Parameters:
        start (list or tuple of size 3): The starting point of the vector.
        end (list or tuple of size 3): The ending point of the vector.

        Returns:
        tuple: The intersection point with the XY plane as (x, y, 0), or None if no intersection.
        """
        x1, y1, z1 = start
        x2, y2, z2 = end

        # If the vector is parallel to the XY plane and not in it, there's no intersection
        if z1 == z2:
            return None

        # Compute the parameter t at which the vector intersects the XY plane (z = 0)
        t = -z1 / (z2 - z1)

        # Calculate the intersection point
        x_intersect = x1 + t * (x2 - x1)
        y_intersect = y1 + t * (y2 - y1)

        return (x_intersect, y_intersect, 0)

    def find_intersection_with_boundaries(self, start, end):
        """
        Find the single intersection point of the ray from start to end with the boundaries of the rectangle.

        Parameters:
        start (tuple): The starting point of the vector.
        end (tuple): The ending point of the vector.
        x_bounds (tuple): The bounds for x values.
        y_bounds (tuple): The bounds for y values.

        Returns:
        tuple: The intersection point with the rectangle boundaries.
        """
        x1, y1, _ = start
        x2, y2, _ = end

        x_bounds = -self.roi_size[0] / 2, self.roi_size[0] / 2
        y_bounds = -self.roi_size[1] / 2, self.roi_size[1] / 2

        # Check intersection with x = x_min
        x_min, x_max = x_bounds
        if x2 != x1:
            t = (x_min - x1) / (x2 - x1)
            if t >= 0:
                y_intersect = y1 + t * (y2 - y1)
                if y_bounds[0] <= y_intersect <= y_bounds[1]:
                    return (x_min, y_intersect, 0)

        # Check intersection with x = x_max
        if x2 != x1:
            t = (x_max - x1) / (x2 - x1)
            if t >= 0:
                y_intersect = y1 + t * (y2 - y1)
                if y_bounds[0] <= y_intersect <= y_bounds[1]:
                    return (x_max, y_intersect, 0)

        # Check intersection with y = y_min
        y_min, y_max = y_bounds
        if y2 != y1:
            t = (y_min - y1) / (y2 - y1)
            if t >= 0:
                x_intersect = x1 + t * (x2 - x1)
                if x_bounds[0] <= x_intersect <= x_bounds[1]:
                    return (x_intersect, y_min, 0)

        # Check intersection with y = y_max
        if y2 != y1:
            t = (y_max - y1) / (y2 - y1)
            if t >= 0:
                x_intersect = x1 + t * (x2 - x1)
                if x_bounds[0] <= x_intersect <= x_bounds[1]:
                    return (x_intersect, y_max, 0)

        return None

    def find_intermediate_corners(self, left_top, right_top):
        x_l, y_l, _ = left_top
        x_r, y_r, _ = right_top
        x_lim, y_lim = self.roi_size[0] / 2, self.roi_size[1] / 2

        # top side: 0, right_side: 1, bot_side: 2, left_side: 3
        assert abs(x_l) == x_lim or abs(y_l) == y_lim
        l_side = (1 if x_l == x_lim else 3) if abs(x_l) == x_lim else (0 if y_l == y_lim else 2)
        r_side = (1 if x_r == x_lim else 3) if abs(x_r) == x_lim else (0 if y_r == y_lim else 2)

        corner_mapping = \
            {
                0: [x_lim, y_lim, 0],
                1: [x_lim, -y_lim, 0],
                2: [-x_lim, -y_lim, 0],
                3: [-x_lim, y_lim, 0],
            }

        intermediate_corners_list = list()

        while l_side != r_side:
            intermediate_corners_list.append(corner_mapping[l_side])
            l_side = (l_side + 1) % 4

        return intermediate_corners_list

    def transform_point(self, point):
        """
        Transform from ego-centric real coordinates (in meters) to array wise coordinates (array indices)
        """
        x_array_roi_prop = (self.bev_array_size[1] - 1) / self.roi_size[0]
        y_array_roi_prop = (self.bev_array_size[0] - 1) / self.roi_size[1]
        x, y = point[0] * x_array_roi_prop, point[1] * y_array_roi_prop
        y = -y
        x = x + ((self.bev_array_size[1] - 1) / 2)
        y = y + ((self.bev_array_size[0] - 1) / 2)
        x = np.round(x).astype(np.int32)
        y = np.round(y).astype(np.int32)
        return x, y  # np.array([x, y], dtype=np.uint8)

    def create_segmentation_map(self, top_left, top_right, bottom_right, bottom_left):
        """
        Create a binary segmentation map.

        Parameters:
        top_left (tuple): The top left point of the area.
        top_right (tuple): The top right point of the area.
        bottom_right (tuple): The bottom right point of the area.
        bottom_left (tuple): The bottom left point of the area.
        map_size (tuple): The size of the map.

        Returns:
        numpy.ndarray: The binary segmentation map.
        """
        # Create an empty map

        seg_map = np.zeros(self.bev_array_size, dtype=np.uint8)
        intermediate_corners = self.find_intermediate_corners(top_left, top_right)

        # Transform the points
        top_left = self.transform_point(top_left)
        top_right = self.transform_point(top_right)
        bottom_right = self.transform_point(bottom_right)
        bottom_left = self.transform_point(bottom_left)

        for idx, pt in enumerate(intermediate_corners):
            intermediate_corners[idx] = self.transform_point(pt)

        # Define the polygon for the smaller area
        points = np.array([top_right, bottom_right, bottom_left, top_left])
        if len(intermediate_corners) != 0:
            intermediate_corners = np.array(intermediate_corners)
            points = np.concatenate((points, intermediate_corners))

        # Create the binary mask for the polygon
        cv2.fillPoly(seg_map, [points], 1)

        # Ensure we get the smaller area by comparing the filled areas
        seg_map_complement = 1 - seg_map
        if np.sum(seg_map) > np.sum(seg_map_complement):
            seg_map = seg_map_complement

        return seg_map

    def find_perspective_indices(self, camera_loc, points_p, corners, grid_x, grid_y):
        A, B, _, C = corners.copy()
        # A = corners[0]
        # B = corners[1]
        # C = corners[3]

        AB = B - A
        AC = C - A
        normal = np.cross(AB, AC)  # Normal vector of the plane

        # Line parameterization for multiple points
        directions = points_p - camera_loc  # Now directions is an array of vectors
        ts = (np.dot(normal, A - camera_loc) / np.dot(directions, normal[:, None])).flatten()
        intersections = camera_loc + ts[:, None] * directions

        # Transform to grid coordinates
        u_x = (B - A) / grid_x
        u_y = (C - A) / grid_y
        u_z = np.cross(u_x, u_y)  # for the sake of matrix to be square,

        matrix = np.array([u_x, u_y, u_z]).T  # Transformation matrix
        constants = intersections - A  # Matrix of vectors to be transformed

        # Solve for k and l for each point
        k_l = np.linalg.solve(matrix, constants.T)  # Transpose constants to match dimensions for solve
        # the values for u_z must be very very close to 0 bc the points are on u_x u_y plane
        assert abs(k_l.T[:, 2]).max() < 1e-6

        return k_l.T[:, :2]


    def generate_grid(self, corners, grid_x, grid_y):
        # Extract corners
        left_top, right_top, right_bottom, left_bottom = corners.copy()

        # Calculate indentation
        indent_w = (right_top - left_top) / (2 * grid_x)
        indent_h = (left_bottom - left_top) / (2 * grid_y)

        # Adjust the corners inward
        left_top += indent_w + indent_h
        right_top -= indent_w - indent_h
        left_bottom += indent_w - indent_h
        right_bottom -= indent_w + indent_h

        # Interpolate along top and bottom edges
        top_edge = np.linspace(left_top, right_top, grid_x)
        bottom_edge = np.linspace(left_bottom, right_bottom, grid_x)

        # Preallocate grid array
        grid = np.empty((grid_y, grid_x, 3))  # Preallocate memory for (grid_y, grid_x, 3)

        # Fill grid with interpolated points
        for j in range(grid_x):  # Iterate along the vertical axis (grid_y)
            grid[:, j] = np.linspace(top_edge[j], bottom_edge[j], grid_y)

        return grid  # Shape: (grid_y, grid_x, 3)


    def find_bev_indices(self,
        camera_loc, points_p, height_level, grid_x, grid_y,
        epsilon=1e-12, fill_invalid=np.nan):
       
        # --- 1. Extract corners and define the plane normal ---
        # Here we assume corners is a Torch tensor, so we call corners.clone().numpy() 
        # if you want a NumPy array; or you can just do corners[0] directly if corners is already np.ndarray.
        # But for simplicity, let's just assume corners is a NumPy array:
        left_top = np.array([-self.roi_size[0] / 2, self.roi_size[1] / 2, height_level])   # Left top
        right_top = np.array([self.roi_size[0] / 2, self.roi_size[1] / 2, height_level])   # Right top
        right_bottom = np.array([self.roi_size[0] / 2, -self.roi_size[1] / 2, height_level])  # Right bottom
        left_bottom = np.array([-self.roi_size[0] / 2, -self.roi_size[1] / 2, height_level])  # Left bottom

        # Combine corners into an array
        corners = np.array([left_top, right_top, right_bottom, left_bottom])
        
        A, B, _, C = corners
        AB = B - A
        AC = C - A
        normal = np.cross(AB, AC)

        # --- 2. Ray parameterization for multiple points ---
        directions = points_p - camera_loc  # shape (N, 3)
        numerator = np.dot(normal, A - camera_loc)  # scalar
        denominator = np.dot(directions, normal)    # shape (N,)

        # --- 3. Compute t = numerator / denominator ---
        #        and check for invalid or behind-camera intersections
        with np.errstate(divide='ignore', invalid='ignore'):
            ts = numerator / denominator  # shape (N,)

        # Valid if:
        #    1) denominator != 0 (abs(denominator) > epsilon)
        #    2) t > 0 => intersection is in front of the camera
        valid_mask = (np.abs(denominator) > epsilon) & (ts > 0)

        # --- 4. Intersection points (for all rays) ---
        # We'll handle validity afterwards
        intersections = camera_loc + directions * ts[:, None]  # shape (N,3)

        # --- 5. Build transform to map intersection -> (k, l) ---
        # Define the basis vectors of the plane in "grid units"
        u_x = AB / grid_x  
        u_y = AC / grid_y
        u_z = np.cross(u_x, u_y)  # so we get a 3x3 invertible matrix

        matrix = np.array([u_x, u_y, u_z]).T  # shape (3,3)

        # We'll solve only for the intersections that are already "valid"
        valid_intersections = intersections[valid_mask] - A  # shape (M, 3)
        k_l_z_valid = np.linalg.solve(matrix, valid_intersections.T).T  # shape (M,3)

        # (k, l) come from the first two columns
        k_valid = k_l_z_valid[:, 0]
        l_valid = k_l_z_valid[:, 1]

        # --- 6. Further refine validity based on 0 <= k <= grid_x and 0 <= l <= grid_y ---
        in_bounds_mask = (
            (k_valid >= 0.0) & (k_valid <= grid_x * 1.01) &
            (l_valid >= 0.0) & (l_valid <= grid_y * 1.01)
        )
        # Combine with the previous valid_mask
        # valid_mask was shape (N,); in_bounds_mask is shape (M,) 
        # for those previously valid. We need to map these M into the correct place in valid_mask.
        # The typical approach is:
        # valid_mask[valid_mask] = in_bounds_mask
        # So that "previously valid" points that are out-of-bounds become invalid now.
        valid_mask_indices = np.flatnonzero(valid_mask)  # indices in [0..N) where valid_mask is True
        valid_mask[valid_mask_indices] = in_bounds_mask

        # Now we can re-extract k, l for the (newly) valid points
        # because some might have been turned invalid if out-of-bounds
        final_valid_indices = np.flatnonzero(valid_mask)
        k_l_z_final = k_l_z_valid[in_bounds_mask]  # shape (final_count, 3)
        k_final = k_l_z_final[:, 0]
        l_final = k_l_z_final[:, 1]

        # --- 7. Prepare the output array ---
        if fill_invalid is None:
            # Option A: Only return the valid (k,l) pairs, all stacked together
            bev_coords = np.column_stack([k_final, l_final])
        else:
            # Option B: Fill invalid intersections with sentinel (np.nan by default)
            bev_coords = np.full((points_p.shape[0], 2), fill_invalid, dtype=np.float32)
            bev_coords[final_valid_indices, 0] = k_final
            bev_coords[final_valid_indices, 1] = l_final

        return bev_coords, valid_mask



    def reverse_transform_indices(self, row_indices, col_indices):
        """
        Shift the indices to the middle of the square to represent a pixel
        Scale the indices to keep up with the roi size
        Invert the rows bc increse in rows means decrease in actual y coordinate
        Columns are the x and rows are y
        Shift the indices to make ego centered instead of top_left centered
        Add a z=0 dimension to represent the ground
        """
        roi_array_prop = self.roi_size[0] / self.bev_array_size[1]
        shift_y = self.roi_size[1] / 2
        shift_x = self.roi_size[0] / 2

        row_indices = row_indices.copy().astype(float)
        col_indices = col_indices.copy().astype(float)

        # row_indices = np.array(row_indices)
        # col_indices = np.array(col_indices)

        row_indices += 0.5
        col_indices += 0.5
        row_indices *= roi_array_prop
        col_indices *= roi_array_prop

        row_indices *= -1

        x = col_indices
        y = row_indices

        x -= shift_x  # new origin is at right (+ direction)
        y += shift_y  # new origin is at below (- direction)

        zeros = np.zeros_like(x)
        points = np.vstack((x, y, zeros)).T
        return points

    def get_bev2perspective_correspondance(self, cam2bev_fov_indices, cam_locs, cam_world_corners,
                                           grid_x, grid_y, height_levels):
        cam2persp_indices = dict()
        cam2bev_distances = dict()
        # bev_on_perspective_indices = dict()

        for level in height_levels.keys():
            cam2persp_indices[level] = dict()
            cam2bev_distances[level] = dict()
            # bev_on_perspective_indices[level] = dict()

        for cam_name, (row_indices, col_indices) in cam2bev_fov_indices.items():
            # find the ego centered coordinates for the array indices (row_indices, col_indices)
            points = self.reverse_transform_indices(row_indices, col_indices)  # convert to roi coordinates
            for str_lvl, val_lvl in height_levels.items():
                level_increaser = np.zeros_like(points)
                level_increaser[:, 2] += val_lvl

                cur_level_points = points + level_increaser
                cam_loc = cam_locs[cam_name]
                world_corners = cam_world_corners[cam_name]
                ks_n_ls = self.find_perspective_indices(cam_loc, cur_level_points, world_corners, grid_x, grid_y) # find perpective indices for each bev point
                
                distances = np.linalg.norm(cur_level_points - cam_loc, axis=1)

                cam2persp_indices[str_lvl][cam_name] = ks_n_ls
                cam2bev_distances[str_lvl][cam_name] = distances  # num_points (not all points, the ones in fov)
                # bev_on_perspective_indices[str_lvl][cam_name] = torch.from_numpy(bev_ks_n_ls)

        return cam2persp_indices, cam2bev_distances # , bev_on_perspective_indices  # a dict[level][cam_name]

    def __call__(self, input_dict: Dict) -> Dict:
        # intrinsics_extrinsics = extract_camera_intrinsics_extrinsics(self.ref_scene, self.ref_frame_idx)
        intrinsics = input_dict["cam_intrinsics"]  # intrinsics = intrinsics_extrinsics["cam_intrinsics"]
        if self.dataset == "nuscenes":
            extrinsics = input_dict["ego2cam"]
        elif self.dataset == "argoverse":
            extrinsics = input_dict["cam_extrinsics"]
        else:
            assert False

        if self.precision > 0:
            # DO NOT compute all the values from scratch, 
            # store the intrinsics, extrinsics and if same, return the alredy computed one
            intr_json = json.dumps([i if isinstance(i, list) else i.tolist() for i in intrinsics], sort_keys=True)
            intr_hash = hashlib.sha256(intr_json.encode()).hexdigest()

            # the translation part may slightly vary so calculate based on the rotation
            rotation_matrices = [mat[:3, :3] for mat in extrinsics]
            rounded_rotation_data = [self.round_matrix(mat) for mat in rotation_matrices]
            extr_json = json.dumps(rounded_rotation_data, sort_keys=True)
            extr_hash = hashlib.sha256(extr_json.encode()).hexdigest()

            # check whether the cam0 translation difference is less than 15cm 
            # if yes return the associated results, if not continue calculation.
            cur_cam0_trans = extrinsics[0][:3, 3] # np array of 3 elments
            if intr_hash in self.historical_results.keys():
                cur_hist_intr = self.historical_results[intr_hash]
                if extr_hash in cur_hist_intr.keys():
                    # shape: list(list(cam0 translation(3), dictionary), *)
                    cur_trans_dict_list = cur_hist_intr[extr_hash]
                    for trans_vec, result_dict in cur_trans_dict_list:
                        # 15cm max 9 not found for 4 gpu 8 loader during test
                        # 10cm max 25 not found for 4 gpu 8 loader during test
                        if np.linalg.norm(cur_cam0_trans - trans_vec) < self.precision:
                            # input_dict["fov_parameters"] = dict()
                            # input_dict["fov_parameters"]["hash_id"] = result_dict["hash_id"]
                            input_dict["fov_parameters"] = copy.deepcopy(result_dict)
                            # print("found")
                            return input_dict
            self.not_found_count += 1
            # print(f"Not found count: {self.not_found_count}")
            # pdb.set_trace()

        if self.dataset == "nuscenes": 
            cam_name_regex = re.compile(r'/CAM_([^/]+)')
            prefix = "CAM_"
        elif self.dataset == "argoverse":
            cam_name_regex = re.compile(r'/ring_([^/]+)')
            prefix = "ring_"
        else:
            assert False

        file_paths = input_dict["img_filenames"]  # file_paths = intrinsics_extrinsics["img_filenames"]
        camera_names = []
        for path in file_paths:
            match = cam_name_regex.search(path)
            if match:
                camera_names.append(prefix + match.group(1))
        num_cam = len(camera_names)
        # pdb.set_trace()
        all_seg_maps = dict()
        cam_locs = dict()
        cam_world_corners = dict()
        # pdb.set_trace()
        for i in range(num_cam):
            # :2 for exclude the channel dim
            image_height, image_width = input_dict["img_shape"][i][:2]
            cur_extrinsics = extrinsics[i]
            cur_intrinsics = intrinsics[i]
            forward_vec, up_vec, cam_location = self.extract_forward_vector(cur_extrinsics)
            # print(forward_vec)
            # print(cam_location)
            world_corners = self.calculate_image_plane_corners(cur_intrinsics, forward_vec, up_vec, cam_location,
                                                               image_width,
                                                               image_height)
            # '106d962b-911d-354d-961d-9abe93119b9c', 
            t_l, t_r, b_r, b_l = world_corners
            b_m_r = (b_r * 9 + t_r) / 10
            b_m_l = (b_l * 9 + t_l) / 10

            top_left_on_plane = self.intersection_with_xy_plane(cam_location, b_m_l)
            top_right_on_plane = self.intersection_with_xy_plane(cam_location, b_m_r)

            bottom_right_on_plane = self.intersection_with_xy_plane(cam_location, b_r)
            bottom_left_on_plane = self.intersection_with_xy_plane(cam_location, b_l)
            # shift the top points to the boundaries
            top_right_on_plane = self.find_intersection_with_boundaries(bottom_right_on_plane, top_right_on_plane)
            top_left_on_plane = self.find_intersection_with_boundaries(bottom_left_on_plane, top_left_on_plane)

            cur_map = self.create_segmentation_map(top_left_on_plane, top_right_on_plane,
                                                   bottom_right_on_plane, bottom_left_on_plane)
            all_seg_maps[camera_names[i]] = cur_map
            # to extract the locations on perspective image (corresponding to the bev)
            cam_locs[camera_names[i]] = cam_location
            cam_world_corners[camera_names[i]] = world_corners

        # pdb.set_trace()
        for key, segmap in all_seg_maps.items():
            indices = np.where(segmap == 1)
            # rows = indices[0]
            # cols = indices[1]
            all_seg_maps[key] = (indices[0], indices[1])

        # perspective indices is the bev array indicating the locations on perspective image
        # bev_indices is the perspective array indicating the locations on bev array
        # perspective_indices, cam2bev_distances, bev_indices = 
        perspective_indices, cam2bev_distances = self.get_bev2perspective_correspondance(all_seg_maps,
                                                                                         cam_locs,
                                                                                         cam_world_corners,
                                                                                         self.perspective_grid_x,
                                                                                         self.perspective_grid_y,
                                                                                         self.height_levels)
        # find the valid indices, even though we extracted the cam_fov, there is still a problem with
        # (probably) poly_fill fn, which might return some indices slightly out of scope of the camera
        # so after we find the corresponding perspective indices (to the bev fov), we refine the indices according
        # to whether the corresponding perspective point is actually located on image
        perspective_valid_mask = dict()
        for cam_name, p_indices in perspective_indices["0"].items():  # take level 0 as reference
            valids = np.logical_and.reduce((
                p_indices[:, 0] <= self.perspective_grid_x,
                p_indices[:, 0] >= 0,
                p_indices[:, 1] <= self.perspective_grid_y,
                p_indices[:, 1] >= 0
            ))
            perspective_valid_mask[cam_name] = valids
            # filter the perspective indices
            # for each level, filter out according to the mask obtained from level 0
            for level in perspective_indices.keys():
                perspective_indices[level][cam_name] = perspective_indices[level][cam_name][valids]
                cam2bev_distances[level][cam_name] = cam2bev_distances[level][cam_name][valids]

        # filter the indices accordingly
        for cam_name, segmap in all_seg_maps.items():
            x_ind, y_ind = all_seg_maps[cam_name]
            x_ind = x_ind[perspective_valid_mask[cam_name]]
            y_ind = y_ind[perspective_valid_mask[cam_name]]
            all_seg_maps[cam_name] = (x_ind, y_ind)

        # handle the overlapping areas
        # Priorities:
        #   Top prior: CAM_FRONT
        #   Mid Prior: CAM_FRONT_RIGHT, CAM_FRONT_LEFT
        #   Low Prior: CAM_BACK_RIGHT, CAM_BACK_LEFT
        #   Not Prior: CAM_BACK
        # First one is prior in the pairs
        if not self.allow_spatial_overlap:
            pairs = [("CAM_FRONT", "CAM_FRONT_RIGHT"), ("CAM_FRONT", "CAM_FRONT_LEFT"),
                     ("CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"), ("CAM_FRONT_LEFT", "CAM_BACK_LEFT"),
                     ("CAM_BACK_LEFT", "CAM_BACK"), ("CAM_BACK_RIGHT", "CAM_BACK")]

            for h_p, l_p in pairs:
                hp_x, hp_y = all_seg_maps[h_p]
                lp_x, lp_y = all_seg_maps[l_p]

                # Combine x and y indices into a tuple of coordinates
                hp_coords = set(zip(hp_x, hp_y))
                lp_coords = list(zip(lp_x, lp_y))

                # Create a mask for lp_coords, where True indicates the pair is not in hp_coords
                lp_mask = np.array([True if coord not in hp_coords else False for coord in lp_coords], dtype=bool)
                lp_x_filtered = lp_x[lp_mask]
                lp_y_filtered = lp_y[lp_mask]

                # Update the low-priority segment in all_seg_maps with the filtered indices
                all_seg_maps[l_p] = (lp_x_filtered, lp_y_filtered)
                # update the persp indices
                # apply the filter for each level
                for level in perspective_indices.keys():
                    perspective_indices[level][l_p] = perspective_indices[level][l_p][lp_mask]
                    cam2bev_distances[level][l_p] = cam2bev_distances[level][l_p][lp_mask]

        # flip the key levels of the perspective indices
        index_map, distance_map = self.convert_to_index_map(perspective_indices, all_seg_maps,
                                                            cam2bev_distances)  # dict of cam name, tensor(num_lvl, h, w, 2) and ... tensor(num_lvl, h, w)
        spatial_mask = self.extract_spatial_mask(all_seg_maps)
        perspective_sparse_distance_mask = self.generate_perspective_distance_mask(index_map, distance_map)  # dict of cam name, tensor(num_lvl, h, w)

        input_dict["fov_parameters"] = dict()
        
        input_dict["fov_parameters"]["index_map"] = index_map
        input_dict["fov_parameters"]["cam_mask"] = spatial_mask
        input_dict["fov_parameters"]["perspective_distance_mask"] = perspective_sparse_distance_mask
        
        if self.precision > 0:
            # for model to find it in its dictionary (cloning is faster than cpu->cuda change)
            rounded_trans = np.round(cur_cam0_trans, decimals=5)
            trans_str = ",".join(map(str, rounded_trans))
            trans_hash = hashlib.sha256(trans_str.encode()).hexdigest()
            hash_id = intr_hash + extr_hash + trans_hash # type: ignore
            input_dict["fov_parameters"]["hash_id"] = hash_id

            self.historical_results[intr_hash] = dict()
            self.historical_results[intr_hash][extr_hash] = list()
            self.historical_results[intr_hash][extr_hash].append([cur_cam0_trans, copy.deepcopy(input_dict["fov_parameters"])])

        return input_dict

    def convert_to_index_map(self, perspective_indices: Dict, array_indices: Dict, distances: Dict):
        # invert the dictionary key levels
        reversed_perspective_indices = {}
        reversed_distances = {}
        # the structure of distances and perspective indices must be same, just the value dimensions differ
        for level, cam_dict in perspective_indices.items():
            for cam_name, value in cam_dict.items():
                if cam_name not in reversed_perspective_indices:
                    reversed_perspective_indices[cam_name] = {}
                    reversed_distances[cam_name] = {}
                reversed_perspective_indices[cam_name][level] = value
                reversed_distances[cam_name][level] = distances[level][cam_name]

        # extract the level order to use it later
        for k, v in reversed_perspective_indices.items():
            sorted_level_names = [l for l in v.keys()]
            sorted_level_names = sorted(sorted_level_names, key=lambda x: int(x))

        index_map = dict()
        all_distance_map = dict()
        for cam, lvl_dict in reversed_perspective_indices.items():
            distance_ls = torch.stack([torch.from_numpy(reversed_distances[cam][l]).float() for l in
                                       sorted_level_names])  # result is (num_level, num_point)
            new_ls = torch.stack([torch.from_numpy(lvl_dict[l]).float() for l in
                                  sorted_level_names])  # result is (num_level, num_point, 2)
            # shift value between -1 and 1 to use it with grid sampler
            new_ls[..., 0] = (new_ls[..., 0] / self.perspective_grid_x) * 2 - 1
            new_ls[..., 1] = (new_ls[..., 1] / self.perspective_grid_y) * 2 - 1

            # construct the index map with indices to use it with grid sampler
            n_lvl, n_point, _ = new_ls.shape

            map = torch.ones((n_lvl, self.bev_h, self.bev_w, 2)) * 99999  # 999 to indicate pad there with 0
            distance_map = torch.zeros((n_lvl, self.bev_h, self.bev_w))

            x_indices_on_array, y_indices_on_array = array_indices[cam][0], array_indices[cam][1]
            map[:, x_indices_on_array, y_indices_on_array] = new_ls
            distance_map[:, x_indices_on_array, y_indices_on_array] = distance_ls

            index_map[cam] = map
            all_distance_map[cam] = distance_map

        return index_map, all_distance_map

    def extract_spatial_mask(self, all_segmap_indices):
        cam_mask = dict()
        for cam, (x_inds, y_inds) in all_segmap_indices.items():
            mask = torch.zeros((self.bev_h, self.bev_w))
            mask[x_inds, y_inds] = 1
            # mask = mask.to(self.device)
            cam_mask[cam] = mask
        return cam_mask

    def generate_perspective_distance_mask(self, index_map, distance_map, threshold=1.5):
        output_mask = {}
        H, W = self.perspective_grid_y, self.perspective_grid_x
        for cam_name, idx_map in index_map.items():
            num_lvl, h, w, _ = idx_map.shape
            mask = torch.zeros((num_lvl, H, W), dtype=torch.float32)
            for lvl in range(num_lvl):
                idx_map_lvl = idx_map[lvl]  # Shape: (h, w, 2)
                dist_map_lvl = distance_map[cam_name][lvl]  # Shape: (h, w)

                # Extract normalized coordinates
                x_norm = idx_map_lvl[..., 0]  # Shape: (h, w)
                y_norm = idx_map_lvl[..., 1]  # Shape: (h, w)

                # Create a valid mask where coordinates are within the threshold
                valid_mask = (x_norm >= -threshold) & (x_norm <= threshold) & \
                             (y_norm >= -threshold) & (y_norm <= threshold)

                # Proceed only with valid indices
                x_valid = x_norm[valid_mask]
                y_valid = y_norm[valid_mask]
                distances_valid = dist_map_lvl[valid_mask]

                # Convert normalized coordinates to pixel coordinates
                x = ((x_valid + 1) / 2) * (W - 1)
                y = ((y_valid + 1) / 2) * (H - 1)

                # Round to integer pixel coordinates
                x_int = torch.round(x).long()
                y_int = torch.round(y).long()

                # Clip to valid pixel indices
                x_int = torch.clamp(x_int, 0, W - 1)
                y_int = torch.clamp(y_int, 0, H - 1)

                # Directly index into the mask (duplicates will overwrite)
                mask[lvl, y_int, x_int] = distances_valid

            output_mask[cam_name] = mask  # Shape: (num_lvl, H, W)

        return output_mask

    def round_matrix(self, matrix, precision=5):
        """Rounds each element in a matrix to a fixed precision while preserving small numbers."""
        return [[float(f"{num:.{precision}g}") for num in row] for row in matrix]
