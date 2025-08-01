import multiprocessing
import os
import glob
import pickle
import time
from typing import List
import warnings

import numpy as np
import torch
import laspy

from datasets.common import PointCloudDataset, grid_subsampling
from utils.config import Config, bcolors
from utils.mayavi_visu import *

class LASDataset(PointCloudDataset):
    """Class to handle LAS-format dataset"""

    def __init__(self, config:Config, set:str="training", 
                 use_potentials:bool=True, load_data:bool=True, 
                 path:str=r"..\..\Data\small_dataset"):
        
        super().__init__("LAS")

        self.label_to_names = {
            1: "unknown",
            2: "ground",
            5: "vegetation",
            6: "building",
            23: "car",
            24: "pole",
            25: "fence",
            26: "powerline",
            27: "truck"
        }

        self.init_labels()

        # classes to be ignored for training
        self.ignored_labels = np.array([])

        # dataset directory
        self.path = path

        # type of task
        self.dataset_task ="cloud_segmentation"

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes - len(self.ignored_labels)
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        # Path of the training files
        self.train_path = "train"
        self.valid_path = "validate"

        # List of files to process
        train_las_path = os.path.join(self.path, self.train_path)
        valid_las_path = os.path.join(self.path, self.valid_path)

        self.all_splits = [0, 1]
        self.validation_split = 1
        
        # Number of models used per epoch
        if self.set == "training":
            self.epoch_n = config.epoch_steps * config.batch_num
        elif self.set in ["validation", "test"]:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError("Unkown set for LAS data: ", self.set)
        
        # Stop data is not needed
        if not load_data:
            return
        
        self.files = []
        if self.set == "training":
            self.files += [os.path.join(train_las_path, f)
                           for f in glob.glob("*.las", root_dir=train_las_path)]
        elif self.set in ["validation", "test", "ERF"]:
            self.files += [os.path.join(valid_las_path, f)
                           for f in glob.glob("*.las", root_dir=valid_las_path)]
            
        if self.set == "training":
            self.cloud_names = [
                os.path.splitext(f)[0]
                for f in glob.glob("*.las", root_dir=train_las_path)
            ]
        elif self.set in ["validation", "test", "ERF"]:
            self.cloud_names = [
                os.path.splitext(f)[0]
                for f in glob.glob("*.las", root_dir=valid_las_path)
            ]

        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError("subsampling_parameter too low: must be over 1cm")
        
        # Initiate containers
        self.input_trees: List[KDTree] = []
        self.input_intensity = []
        self.input_labels = []
        self.pot_trees: List[KDTree] = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []

        # Start loading
        self.load_subsampled_clouds()

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [
                    torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)
                ]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(
                np.array(self.argmin_potentials, dtype=np.int64)
            )
            self.min_potentials = torch.from_numpy(
                np.array(self.min_potentials, dtype=np.float64)
            )
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch.tensor(
                [0 for _ in range(config.input_threads)], dtype=torch.int32
            )
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            self.epoch_inds = torch.from_numpy(
                np.zeros((2, self.epoch_n), dtype=np.int64)
            )
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = multiprocessing.Lock()

        # For ERF visualization, we want only one cloud per batch and no 
        # randomness
        if self.set == "ERF":
            self.batch_limit = torch.tensor([1], dtype=torch.float32)
            self.batch_limit.share_memory_()
            np.random.seed(42)

        return

    def __getitem__(self, idx) -> list:
        """The main thread gives a list of indices to load a batch.
        Each worker is going to work in parallel to load a different
        list of indices.
        """
        if self.use_potentials:
            return self.potential_item()
        else:
            return self.random_item()

    def potential_item(self, debug_workers=False):
        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0
        failed_attempts = 0

        info = torch.utils.data.get_worker_info()
        if info is not None:
            wid = info.id
        else:
            wid = None

        while True:
            t += [time.time()]

            if debug_workers:
                message = ""
                for wi in range(info.num_workers):
                    if wi == wid:
                        message += f" {bcolors.FAIL}X{bcolors.ENDC} "
                    elif self.worker_waiting[wi] == 0:
                        message += "   "
                    elif self.worker_waiting[wi] == 1:
                        message += " | "
                    elif self.worker_waiting[wi] == 2:
                        message += " o "
                print(message)
                self.worker_waiting[wid] = 0

            with self.worker_lock:
                if debug_workers:
                    message = ""
                    for wi in range(info.num_workers):
                        if wi == wid:
                            message += f" {bcolors.FAIL}v{bcolors.ENDC} "
                        elif self.worker_waiting[wi] == 0:
                            message += "   "
                        elif self.worker_waiting[wi] == 1:
                            message += " | "
                        elif self.worker_waiting[wi] == 2:
                            message += " o "
                    print(message)
                    self.worker_waiting[wid] = 1

                # Get potential minimum
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])

                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, 
                                      copy=False)

                # Center point of input region
                center_point = np.copy(pot_points[point_ind, :].reshape(1, -1))

                # Add small noise to center point
                if self.set != "ERF":
                    center_point += np.clip(
                        np.random.normal(
                            scale = self.config.in_radius / 10,
                            size=center_point.shape
                        ),
                        -self.config.in_radius / 2,
                        self.config.in_radius / 2,
                    )

                # Indices of points in input region
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(
                    center_point,
                    r=self.config.in_radius,
                    return_distance=True
                )

                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]

                # Update potentials (Tukey weights)
                if self.set != "ERF":
                    tukeys = np.square(
                        1 - d2s / np.square(self.config.in_radius)
                    )
                    tukeys[d2s > np.square(self.config.in_radius)] = 0
                    self.potentials[cloud_ind][pot_inds] += tukeys
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = \
                        self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(
                center_point,
                r=self.config.in_radius
            )[0]

            t += [time.time()]

            # Number collected
            n = input_inds.shape[0]

            # Safe check for empty spheres
            if n < 2:
                failed_attempts += 1
                if failed_attempts > 100 * self.config.batch_num:
                    raise ValueError("Too many empty spheres")
                t += [time.time()]
                t += [time.time()]
                continue

            # Collect labels and intensity
            input_points = (points[input_inds] - center_point)
            input_points = input_points.astype(np.float32)
            input_intensity = self.input_intensity[cloud_ind][input_inds]
            if len(input_intensity.shape) == 1:
                input_intensity = np.expand_dims(input_intensity, axis=1)
            if self.set in ["test", "ERF"]:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]

            t += [time.time()]

            # Data augemntation
            input_points, scale, R = self.augmentation_transform(
                input_points
            )

            # Intensity augmentation
            if np.random.rand() > self.config.augment_color:
                input_intensity *= 0

            # Get original height as additional feature
            input_features = np.hstack(
                (input_intensity, input_points[:, 2:] + center_point[:, 2:])
            ).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 2:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 3:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError("Only accepted input dimensions are 1, 2 and 3 "
                             "(without and with XYZ)")
        
        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs(
            stacked_points, stacked_features, labels, stack_lengths
        )

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        if debug_workers:
            message = ""
            for wi in range(info.num_workers):
                if wi == wid:
                    message += " {:}0{:} ".format(bcolors.OKBLUE, bcolors.ENDC)
                elif self.worker_waiting[wi] == 0:
                    message += "   "
                elif self.worker_waiting[wi] == 1:
                    message += " | "
                elif self.worker_waiting[wi] == 2:
                    message += " o "
            print(message)
            self.worker_waiting[wid] = 2

        t += [time.time()]

        # Display timings
        debugT = False
        if debugT:
            print("\n************************\n")
            print("Timings:")
            ti = 0
            N = 5
            mess = "Init ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + N * i + 1] - t[ti + N * i])
                for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += " {:5.1f}".format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Pots ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + N * i + 1] - t[ti + N * i])
                for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += " {:5.1f}".format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Sphere .... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + N * i + 1] - t[ti + N * i])
                for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += " {:5.1f}".format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Collect ... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + N * i + 1] - t[ti + N * i])
                for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += " {:5.1f}".format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Augment ... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + N * i + 1] - t[ti + N * i])
                for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += " {:5.1f}".format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += N * (len(stack_lengths) - 1) + 1
            print("concat .... {:5.1f}ms".format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print("input ..... {:5.1f}ms".format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print("stack ..... {:5.1f}ms".format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print("\n************************\n")
        return input_list
    
    def random_item(self):
        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0
        failed_attempts = 0

        while True:
            with self.worker_lock:
                # Get potential minimum
                cloud_ind = int(self.epoch_inds[0, self.epoch_i])
                point_ind = int(self.epoch_inds[1, self.epoch_i])

                # Update epoch indices
                self.epoch_i += 1
                if self.epoch_i >= int(self.epoch_inds.shape[1]):
                    self.epoch_i -= int(self.epoch_inds.shape[1])

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Center point of input region
            center_point = np.copy(points[point_ind, :].reshape(1, -1))

            # Add a small noise to center point
            if self.set != "ERF":
                center_point += np.clip(
                    np.random.normal(
                        scale=self.config.in_radius / 10, 
                        size=center_point.shape
                    ),
                    -self.config.in_radius / 2,
                    self.config.in_radius / 2,
                )

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(
                center_point, r=self.config.in_radius
            )[0]

            # Number collected
            n = input_inds.shape[0]

            # Safe check for empty spheres
            if n < 2:
                failed_attempts += 1
                if failed_attempts > 100 * self.config.batch_num:
                    raise ValueError("Too many empty input spheres")
                continue

            # Collect labels and intensity
            input_points = (points[input_inds] - center_point)
            input_points = input_points.astype(np.float32)
            input_intensity = self.input_intensity[cloud_ind][input_inds]
            if self.set in ["test", "ERF"]:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Intensity augmentation
            if np.random.rand() > self.config.augment_color:
                input_intensity *= 0

            # Get original height as additional feature
            input_features = np.hstack(
                (input_intensity, input_points[:, 2:] + center_point[:, 2:])
            ).astype(np.float32)

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 2:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 3:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError("Only accepted input dimensions are 1, 2 and 3 "
                             "(without and with XYZ)")
        
        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(
            stacked_points, stacked_features, labels, stack_lengths
        )

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        return input_list

    def load_subsampled_clouds(self):
        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = os.path.join(self.path, "input_{:.3f}".format(dl))
        if not os.path.isdir(tree_path):
            os.makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]

            # Name of the input files
            KDTree_file = os.path.join(tree_path, f"{cloud_name:s}_KDTree.pkl")
            sub_las_file = os.path.join(tree_path, f"{cloud_name:s}.las")

            # Check if inputs have already been computed
            if os.path.isfile(KDTree_file):
                print(f"\nFound KDTree for cloud {cloud_name:s}, subsampled at "
                      f"{dl:.3f}")

                # Read las into data
                with laspy.open(sub_las_file) as f:
                    data = f.read()
                    sub_intensity = data["intensity"]
                    sub_labels = np.array(data["classification"], dtype=np.int32)

                # Read pkl into search_tree
                with open(KDTree_file, "rb") as f:
                    search_tree = pickle.load(f)

            else:
                print(f"\nPreparing KDTree for cloud {cloud_name:s}, "
                      f"subsampled at {dl:.3f}")

                # Read las file
                with laspy.open(file_path) as f:
                    data = f.read()
                    points = data.xyz.astype(np.float32)
                    intensity = data["intensity"]
                # "unsqueeze" intensity
                intensity = np.expand_dims(intensity, axis=1)
                labels = np.array(data["classification"], dtype=np.int32)

                # Subsample cloud
                sub_points, sub_intensity, sub_labels = grid_subsampling(
                    points, features=intensity, labels=labels, sampleDl=dl
                )

                # Normalize intensity and squeeze label
                # Intensity is 16-bit unisgned integer so divide by max value
                sub_intensity = sub_intensity / 0xFFFF
                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10) 

                # Save KDTree
                with open(KDTree_file, "wb") as f:
                    pickle.dump(search_tree, f)

                # Save las
                sub_data = laspy.create(point_format=6, file_version="1.4")
                sub_data["x"] = sub_points[:,0]
                sub_data["y"] = sub_points[:,1]
                sub_data["z"] = sub_points[:,2]
                sub_data["intensity"] = np.squeeze(sub_intensity)
                sub_data["classification"] = sub_labels
                sub_data.write(sub_las_file)

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_intensity += [sub_intensity]
            self.input_labels += [sub_labels]

            # Number of points x 30 bytes per point record
            size = sub_intensity.shape[0] * 30 / (1024 ** 2)
            t1 = time.time() - t0
            print(f"{size:.1f} MB loaded in {t1:.1f}s")

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print("\nPreparing potentials")

            #Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            for i, file_path in enumerate(self.files):
                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = os.path.join(
                    tree_path, f"{cloud_name}_coarse_KDTree.pkl"
                )

                # Check if inputs have already been computed
                if os.path.isfile(coarse_KDTree_file):
                    # Read pkl to search_tree
                    with open(coarse_KDTree_file, "rb") as f:
                        search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data,
                                          copy=False)
                    coarse_points = grid_subsampling(
                        sub_points.astype(np.float32), sampleDl=pot_dl
                    )

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, "wb") as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]
                cloud_ind += 1

            t1 = time.time() - t0
            print(f"Done in {t1:.1f}s")

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ["validation", "test"]:
            print("\nPreparing reprojection indices for testing")

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):
                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                #File name for saving
                proj_file = os.path.join(tree_path, f"{cloud_name}_proj.pkl")

                # Try to load previous indices
                if os.path.isfile(proj_file):
                    with open(proj_file, "rb") as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    with laspy.open(file_path) as f:
                        data = f.read()
                        points = data.xyz
                        labels = np.array(data["classification"], dtype=np.int32)

                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, 
                                                     return_distance=False)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, "wb") as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                t1 = time.time() - t0
                print(f"{cloud_name} done in {t1:.1f}s")

        print()
        return
    
    def load_evaluation_points(self, file_path):
        """Load points (from test or validation split) on which the
        metrics should be evaluated
        """
        # Get original points
        with laspy.open(file_path) as f:
            data = f.read()
            return data.xyz
    
# ------------------------------------------------------------------------------
#           Utility classes definition
#       \********************************/

class LASSampler(torch.utils.data.Sampler):
    """Sampler for LAS-format datasets"""

    def __init__(self, dataset: LASDataset):
        super().__init__(dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of steps per epoch
        if dataset.set == "training":
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return
    
    def __iter__(self):
        """Yield next batch indices.
        
        In this dataset, this is a dummy sampler that yields the index
        of of the batch element (input sphere) in epoch instead of the
        list of point indices.
        """
        if not self.dataset.use_potentials:
            # Initiate current epoch index
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int64)

            # Number of sphere centers taken per class in each cloud
            num_centers = self.N * self.dataset.config.batch_num
            random_pick_n = int(np.ceil(
                num_centers / self.dataset.config.num_classes
            ))

            # Choose random points of each class for each cloud
            for label_ind, label in enumerate(self.dataset.label_values):
                if label not in self.dataset.ignored_labels:
                    # Gether indices of the points with this label in all the
                    # input clouds
                    all_label_indices = []
                    for cloud_ind, cloud_labels in enumerate(
                        self.dataset.input_labels
                    ):
                        label_indices = np.where(
                            np.equal(cloud_labels, label)
                        )[0]
                        all_label_indices.append(
                            np.vstack((
                                np.full(label_indices.shape, cloud_ind, 
                                        dtype=np.int64),
                                label_indices,
                            ))
                        )

                    all_label_indices = np.hstack(all_label_indices)

                    # Select a random index
                    N_inds = all_label_indices.shape[1]
                    if N_inds < random_pick_n:
                        chosen_label_inds = np.zeros((2, 0), dtype=np.int64)
                        while chosen_label_inds.shape[1] < random_pick_n:
                            chosen_label_inds = np.hstack((
                                chosen_label_inds,
                                all_label_indices[
                                    :, 
                                    np.random.permutation(N_inds)
                                ]
                            ))
                        warnings.warn("When choosing random indices "
                                      "(use_potentials=False), class "
                                      f"{label:d}: "
                                      f"{self.dataset.label_names[label_ind]} "
                                      f"only had {N_inds:d} available points, "
                                      f"while we needed {random_pick_n:d}. "
                                      "Repeating indices in the same epoch.")
                    
                    elif N_inds < 50 * random_pick_n:
                        rand_inds = np.random.choice(
                            N_inds, size=random_pick_n, replace=False
                        )
                        chosen_label_inds = all_label_indices[:, rand_inds]

                    else:
                        chosen_label_inds = np.zeros((2, 0), dtype=np.int64)
                        while chosen_label_inds.shape[1] < random_pick_n:
                            rand_inds = np.unique(np.random.choice(
                                N_inds,
                                size=2 * random_pick_n,
                                replace=True
                            ))
                            chosen_label_inds = np.hstack((
                                chosen_label_inds,
                                all_label_indices[:, :random_pick_n]
                            ))
                        chosen_label_inds = chosen_label_inds[:, :random_pick_n]
                    
                    # Stack for each label
                    all_epoch_inds = np.hstack((
                        all_epoch_inds,
                        chosen_label_inds
                    ))

            # Random permutation of the indices
            random_order = np.random.permutation(all_epoch_inds.shape[1])
            random_order = random_order[:num_centers]
            all_epoch_inds = all_epoch_inds[:, random_order].astype(np.int64)

            # Update epoch inds
            self.dataset.epoch_inds += torch.from_numpy(all_epoch_inds)

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """The number of yielded samples is variable."""
        return self.N
    
    def fast_calib(self):
        """
        This method calibrates the batch sizes while ensuring the 
        potentials are well initialized. Indeed on a dataset like 
        Semantic3D, before potentials have been updated over the 
        dataset, there is a chance that all the dense areas are picked 
        in the begining and in the end, then we will have a very large 
        batch of small point clouds.
        """
        # Estimated average batch size and target value
        estim_b = 0
        target_b = self.dataset.config.batch_num

        # Calibration parameters
        low_pass_T = 10
        Kp = 100.0
        finer = False
        breaking = False

        # Convergence parameters
        smooth_errors = []
        converge_threshold = 0.1

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(2)

        for epoch in range(10):
            for i, test in enumerate(self):

                # New time
                t = t[-1:]
                t += [time.time()]

                # batch length
                b = len(test)

                # Update estim_b (low pass filter)
                estim_b += (b - estim_b) / low_pass_T

                # Estimate error (noisy)
                error = target_b - b

                # Save smooth errors for convergene check
                smooth_errors.append(target_b - estim_b)
                if len(smooth_errors) > 10:
                    smooth_errors = smooth_errors[1:]

                # Update batch limit with P controller
                self.dataset.batch_limit += Kp * error

                # finer low pass filter when closing in
                if not finer and np.abs(estim_b - target_b) < 1:
                    low_pass_T = 100
                    finer = True

                # Convergence
                if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                    breaking = True
                    break

                # Average timing
                t += [time.time()]
                mean_dt = (0.9 * mean_dt 
                           + 0.1 * (np.array(t[1:]) - np.array(t[:-1])))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    print(
                        f"Step {i:5d} estim_b = {estim_b:5.2f} "
                        f"batch_limit = {int(self.dataset.batch_limit):7d}, "
                        f" //  {(1000 * mean_dt[0]):.1f}ms "
                        f" {(1000 * mean_dt[1]):.1f}ms"
                    )

            if breaking:
                break

    def calibration(self, dataloader:torch.utils.data.DataLoader, 
                    untouched_ratio=0.9, verbose=False, force_redo=False):
        """Method performing batch and neighbour calibration.
        
        Batch calibration: Set `batch_limit` (the maximum number of
        points allowed in every batch) so that the average batch
        size (number of stacked point clouds) is as specified.
        
        Neighbour calibration: Set the `neighborhood_limits` (the 
        maximum number of neighbours allowed in convolutions) so
        that 90% of the neighbourhoods remain untouched.
        """
        ##############################
        # Previously saved calibration
        ##############################

        print("\nStarting Calibration (use verbose=True for more details)")
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, "batch_limits.pkl")
        if os.path.isfile(batch_lim_file):
            with open(batch_lim_file, "rb") as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.use_potentials:
            sampler_method = "potentials"
        else:
            sampler_method = "random"
        key = (f"{sampler_method}_{self.dataset.config.in_radius:.3f}_"
               f"{self.dataset.config.first_subsampling_dl:.3f}_"
               f"{self.dataset.config.batch_num:d}")
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print("\nPrevious calibration found:")
            print("Check batch limit dictionary")
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = "?"
            print(f'{color}"{key}": {v}{bcolors.ENDC}')

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, "neighbors_limits.pkl")
        if os.path.isfile(neighb_lim_file):
            with open(neighb_lim_file, "rb") as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists 
        # (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):
            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = f"{dl:.3f}_{r:.3f}"
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print("Check neighbors limit dictionary")
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = f"{dl:.3f}_{r:.3f}"

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = "?"
                print(f'{color}"{key}": {v}{bcolors.ENDC}')

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors 
            # number in a neighborhood
            hist_n = int(np.ceil(4 
                                 / 3 
                                 * np.pi 
                                 * (self.dataset.config.deform_radius + 1) 
                                 ** 3))
            
            # Histogram of neighborhood sizes
            neighb_hists = np.zeros(
                (self.dataset.config.num_layers, hist_n), 
                dtype=np.int32
            )

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num

            # Expected batch size order of magnitude
            expected_N = 10000

            # Calibration parameters. Higher means faster but can also 
            # become unstable. Reduce Kp and Kd if your GPU is small as 
            # the total number of points per batch will be smaller
            low_pass_T = 100
            Kp = expected_N / 200
            Ki = 0.001 * Kp
            Kd = 5 * Kp
            finer = False
            stabilized = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False
            error_I = 0
            error_D = 0
            last_error = 0

            debug_in = []
            debug_out = []
            debug_b = []
            debug_estim_b = []

            #####################
            # Perform calibration
            #####################

            # number of batch per epoch
            sample_batches = 999
            for epoch in range((sample_batches // self.N) + 1):
                for batch_i, batch in enumerate(dataloader):
                    batch:LASCustomBatch
                    # Update neighborhood histogram
                    counts = [
                        np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1)
                        for neighb_mat in batch.neighbors
                    ]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] 
                             for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.cloud_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b
                    error_I += error
                    error_D = error - last_error
                    last_error = error

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 30:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit += (Kp * error 
                                                 + Ki * error_I 
                                                 + Kd * error_D)

                    # Instability detection
                    if not stabilized and self.dataset.batch_limit < 0:
                        Kp *= 0.1
                        Ki *= 0.1
                        Kd *= 0.1
                        stabilized = True

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if (finer 
                        and np.max(np.abs(smooth_errors)) < converge_threshold):
                        
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        print(f"Step {i:5d} estim_b = {estim_b:5.2f} "
                              "batch_limit = "
                              f"{int(self.dataset.batch_limit):7d}")

                    # Debug plots
                    debug_in.append(int(batch.points[0].shape[0]))
                    debug_out.append(int(self.dataset.batch_limit))
                    debug_b.append(b)
                    debug_estim_b.append(estim_b)

                if breaking:
                    break

            # Plot in case we did not reach convergence
            if not breaking:
                import matplotlib.pyplot as plt

                print("ERROR: It seems that the calibration has not reached "
                      "convergence. Here are some plots to understand why:")
                print("If you notice instability, reduce the expected_N value")
                print("If convergece is too slow, increase the expected_N "
                      "value")

                plt.figure()
                plt.plot(debug_in)
                plt.plot(debug_out)

                plt.figure()
                plt.plot(debug_b)
                plt.plot(debug_estim_b)

                plt.show()

                exit(1)

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(
                cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), 
                axis=0
            )
            self.dataset.neighborhood_limits = percentiles

            if verbose:
                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print("\n**************************************************\n")
                line0 = "neighbors_num "
                for layer in range(neighb_hists.shape[0]):
                    line0 += f"|  layer {layer:2d}  "
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = f"     {neighb_size:4d}     "
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += (
                            f"|{color}{neighb_hists[layer, neighb_size]}"
                            f"{bcolors.ENDC}  "
                        )

                    print(line0)

                print("\n**************************************************\n")
                print("\nchosen neighbors limits: ", percentiles)
                print()

            # Save batch_limit dictionary
            if self.dataset.use_potentials:
                sampler_method = "potentials"
            else:
                sampler_method = "random"
            key = (f"{sampler_method}_{self.dataset.config.in_radius:.3f}_"
                   f"{self.dataset.config.first_subsampling_dl:.3f}_"
                   f"{self.dataset.config.batch_num:d}")
            batch_lim_dict[key] = float(self.dataset.batch_limit)
            with open(batch_lim_file, "wb") as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = f"{dl:.3f}_{r:.3f}"
                neighb_lim_dict[key] = \
                    self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, "wb") as file:
                pickle.dump(neighb_lim_dict, file)

        t1 = time.time() - t0
        print(f"Calibration done in {t1:.1f}s\n")
        return

class LASCustomBatch:
    """Custom batch definition with memory pinning for LAS data"""
    def __init__(self, input_list):
        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 7) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [
            torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]
        ]
        ind += L
        self.neighbors = [
            torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]
        ]
        ind += L
        self.pools = [
            torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]
        ]
        ind += L
        self.upsamples = [
            torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]
        ]
        ind += L
        self.lengths = [
            torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]
        ]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind]).long()
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

        return
    
    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

        return self
    
    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

        return self
    
    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements("points", layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements("neighbors", layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements("pools", layer)
    
    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """Return a list of the stacked elements in the batch at a certain 
        layer. If no layer is given, then return all layers
        """
        if element_name == "points":
            elements = self.points
        elif element_name == "neighbors":
            elements = self.neighbors
        elif element_name == "pools":
            elements = self.pools[:-1]
        else:
            raise ValueError("Unknown element name: {:s}".format(element_name))
        
        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == "pools":
                    lengths = self.lengths[layer_i + 1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0 : i0 + length]
                    if element_name == "neighbors":
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == "pools":
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(
                            self.lengths[layer_i][:b_i]
                        )
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list
    
def LASCollate(batch_data):
    return LASCustomBatch(batch_data)

# ------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/

def debug_upsampling(dataset:LASDataset, loader:torch.utils.data.DataLoader):
    """Show which labels are sampled according to strategy chosen"""
    for epoch in range(10):
        for batch_i, batch in enumerate(loader):
            batch:LASCustomBatch
            pc1 = batch.points[1].numpy()
            pc2 = batch.points[2].numpy()
            up1 = batch.upsamples[1].numpy()

            print(pc1.shape, "=>", pc2.shape)
            print(up1.shape, np.max(up1))

            pc2 = np.vstack((pc2, np.zeros_like(pc2[:1, :])))

            # Get neighbors distance
            p0 = pc1[10, :]
            neighbs0 = up1[10, :]
            neighbs0 = pc2[neighbs0, :] - p0
            d2 = np.sum(neighbs0**2, axis=1)

            print(neighbs0.shape)
            print(neighbs0[:5])
            print(d2[:5])

            print("******************")
        print("*******************************************")

    flat_labels = (np.concatenate(dataset.input_labels) 
                   if isinstance(dataset.input_labels[0], (list, np.ndarray)) 
                   else dataset.input_labels)
    _, counts = np.unique(flat_labels, return_counts=True)
    print(counts)

def debug_timing(dataset:LASDataset, loader):
    """Timing of generator function"""
    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config.batch_num
    estim_N = 0

    for epoch in range(10):
        for batch_i, batch in enumerate(loader):
            batch:LASCustomBatch
            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.cloud_inds) - estim_b) / 100
            estim_N += (batch.features.shape[0] - estim_N) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                print(
                    f"Step {batch_i:08d} -> (ms/batch) "
                    f"{(1000 * mean_dt[0]):8.2f} {(1000 * mean_dt[1]):8.2f} "
                    f"/ batch = {estim_b:.2f} - {estim_N:.0f}"
                )

        print("************* Epoch ended *************")
        
    flat_labels = (np.concatenate(dataset.input_labels) 
                   if isinstance(dataset.input_labels[0], (list, np.ndarray)) 
                   else dataset.input_labels)
    _, counts = np.unique(flat_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset:LASDataset, loader):
    for epoch in range(10):
        L = dataset.config.num_layers
        
        for batch_i, batch in enumerate(loader):
            batch:LASCustomBatch
            # Print characteristics of input tensors
            print("\nPoints tensors")
            for i in range(L):
                print(batch.points[i].dtype, batch.points[i].shape)
            print("\nNeigbors tensors")
            for i in range(L):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print("\nPools tensors")
            for i in range(L):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print("\nStack lengths")
            for i in range(L):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print("\nFeatures")
            print(batch.features.dtype, batch.features.shape)
            print("\nLabels")
            print(batch.labels.dtype, batch.labels.shape)
            print("\nAugment Scales")
            print(batch.scales.dtype, batch.scales.shape)
            print("\nAugment Rotations")
            print(batch.rots.dtype, batch.rots.shape)
            # print("\nModel indices")
            # print(batch.model_inds.dtype, batch.model_inds.shape)

            print("\nAre input tensors pinned")
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            # print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print("*******************************************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset:LASDataset, loader):
    """Timing of generator function"""
    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for epoch in range(10):
        for batch_i, input_list in enumerate(loader):
            # New time
            t = t[-1:]
            t += [time.time()]

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                print(
                    f"Step {batch_i:08d} -> Average timings (ms/batch) "
                    f"{(1000 * mean_dt[0]):8.2f} {(1000 * mean_dt[1]):8.2f} "
                )

        print("************* Epoch ended *************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)