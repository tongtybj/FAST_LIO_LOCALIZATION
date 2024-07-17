#!/usr/bin/env python
# coding=utf8
from __future__ import print_function, division, absolute_import

import copy
import time

import open3d as o3d
if int(o3d.__version__.split('.')[1]) < 10:
    o3d_registration = o3d.registration
else:
    o3d_registration = o3d.pipelines.registration

import rospy
import ros_numpy
from geometry_msgs.msg import PoseStamped, Pose, Point, Vector3, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import numpy as np
import tf
import tf.transformations
import tf2_ros

np.set_printoptions(precision=3, suppress=True)

class GlobalLocalization():
    def __init__(self):

        self.converge_cnt = 0
        self.receive_new_scan = False


        self.global_maps = None
        self.cur_odom = None
        self.cur_scan = None
        self.T_map_to_odom = np.eye(4)

        self.stack_cnt = 0
        self.stack_scan = None

        # basic parameter
        self.localization_freq = rospy.get_param("~localization_freq", 0.5)
        self.oneshot = rospy.get_param("~oneshot", False)
        self.headless = rospy.get_param("~headless", False)
        self.reverse_tf = rospy.get_param("~reverse_tf", False)

        # get init frame offset from rosparam

        init_pose_ns = "~initial_guess"
        xyz = rospy.get_param(init_pose_ns + "/pos", [0, 0, 0])
        rpy = rospy.get_param(init_pose_ns + "/rpy", [0, 0, 0])
        init_pos = Point(*xyz)
        init_q = Quaternion(*tf.transformations.quaternion_from_euler(*rpy))
        self.T_map_to_odom =  np.matmul(tf.listener.xyz_to_mat44(init_pos),
                                  tf.listener.xyzw_to_mat44(init_q))

        # parameter for registration
        registration_ns = "~registration"
        self.scan_stack_size = rospy.get_param(registration_ns + "/scan_stack_size", 10)

        self.phase_list = ["init", "float", "fix"]
        self.phase_parameters = {}
        for phase in self.phase_list:
            ns = registration_ns + "/" + phase
            params = {}
            params["map_voxel_size"] = rospy.get_param(ns + "/map_voxel_size", 0.4)
            params["scan_voxel_size"] = rospy.get_param(ns + "/scan_voxel_size", 0.1)
            params["max_corres_dist"] = rospy.get_param(ns + "/max_corres_dist", 1.0)
            params["localization_thresh"] = rospy.get_param(ns + "/localization_thresh", 0.8)
            params["map_range"] = rospy.get_param(ns + "/map_range", 10.0)
            params["scan_range"] = rospy.get_param(ns + "/scan_range", 5.0)
            self.phase_parameters[phase] = params
        self.phase = 0

        # publishers
        self.pub_pc_in_map = rospy.Publisher('/cur_scan_in_map', PointCloud2, queue_size=1)
        self.pub_submap = rospy.Publisher('/submap', PointCloud2, queue_size=1)
        self.pub_map_to_odom = rospy.Publisher('/map_to_odom', Odometry, queue_size=1)
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        # subscriver
        rospy.Subscriber('/cloud_registered', PointCloud2, self.cb_cur_scan, queue_size=1)
        rospy.Subscriber('/Odometry', Odometry, self.cb_cur_odom, queue_size=1)

        rospy.logwarn('Waiting for global map......')
        self.initialize_global_map(rospy.wait_for_message('/3d_map', PointCloud2))
        rospy.loginfo('Get global map......')

        while self.phase == 0:

            if self.receive_new_scan:
                self.global_localization()
            else:
                rospy.logwarn_throttle(1.0, 'Waiting for current scan data')

            rospy.sleep(0.05)

        rospy.loginfo('Initialize successfully!!!!!! \n')

        if not self.oneshot:

            rospy.Timer(rospy.Duration(1 / self.localization_freq), self.thread_localization)


    def initialize_global_map(self, pc_msg):

        raw_global_map = o3d.geometry.PointCloud()
        raw_global_map.points = o3d.utility.Vector3dVector(self.msg_to_array(pc_msg)[:, :3])

        rospy.loginfo('Global map received.')

        self.global_maps = {}
        for k, v in self.phase_parameters.items():
            self.global_maps[k] = self.voxel_down_sample(raw_global_map, v["map_voxel_size"])
            rospy.loginfo('Global map stored {}.'.format(k))

        rospy.loginfo('All Global map stored.')

    def thread_localization(self, msg):

        if not self.receive_new_scan:

            rospy.sleep(0.01)

            return

        self.global_localization()


    def global_localization(self):

        rospy.loginfo('Global localization by scan-to-map matching......')

        tic = time.time()

        crop_scan = self.crop_curr_scan()
        crop_global_map = self.crop_global_map()

        phase_name = self.phase_list[self.phase]

        prev_transform = self.T_map_to_odom
        if self.phase == 0:

            # rough ICP point matching for initialize phase
            prev_transform, fitness = self.registration_at_scale(crop_scan, crop_global_map, \
                                                                 prev_transform, \
                                                                 max_iteration = 1000)

            toc = time.time()
            print("rough fitness: {}, time: {}".format(fitness, toc - tic))

        transform, fitness = self.registration_at_scale(crop_scan, crop_global_map, \
                                                        prev_transform)

        toc = time.time()
        rospy.loginfo('Time: {:.2f}; fitness score:{:.2f}'.format(toc - tic, fitness))

        self.receive_new_scan = False

        phase_name = self.phase_list[self.phase]

        thresh = self.phase_parameters[phase_name]["localization_thresh"]

        if fitness < thresh:

            rospy.logwarn('Not valid matching in phase {}'.format(phase_name))

            self.phase -= 1
            if self.phase < 0:
                self.phase = 0
            self.converge_cnt = 0

            return

        if self.phase == 0:
            # intialize phase
            self.phase += 1
        elif self.phase == 1:
            # float phase
            thresh = self.phase_parameters["fix"]["localization_thresh"]

            if fitness > thresh:
                self.converge_cnt +=1
                if self.converge_cnt > 5:
                    self.phase += 1
                    rospy.loginfo('shift to fix mode')
        else:
            # fix pahse
            pass

        self.T_map_to_odom = transform

        # publish map_to_odom and tf
        map_to_odom = Odometry()
        pos = tf.transformations.translation_from_matrix(self.T_map_to_odom)
        quat = tf.transformations.quaternion_from_matrix(self.T_map_to_odom)
        euler = np.array(tf.transformations.euler_from_quaternion(quat))

        map_to_odom.pose.pose = Pose(Point(*pos), Quaternion(*quat))
        map_to_odom.header.stamp = self.cur_odom.header.stamp
        map_to_odom.header.frame_id = 'map'
        self.pub_map_to_odom.publish(map_to_odom)

        static_transformStamped = TransformStamped()
        static_transformStamped.header.stamp = self.cur_odom.header.stamp
        static_transformStamped.header.frame_id = 'map'
        static_transformStamped.child_frame_id = 'camera_init'
        static_transformStamped.transform.translation = Vector3(*pos)
        static_transformStamped.transform.rotation = Quaternion(*quat)

        rospy.loginfo("Map to Odom: pos: {}, euler: {} \n".format(pos, euler))

        if self.reverse_tf:
            static_transformStamped.header.frame_id = 'camera_init'
            static_transformStamped.child_frame_id = 'map'
            T_odom_to_map = tf.transformations.inverse_matrix(self.T_map_to_odom)
            pos = tf.transformations.translation_from_matrix(T_odom_to_map)
            static_transformStamped.transform.translation = Vector3(*pos)
            quat = tf.transformations.quaternion_from_matrix(T_odom_to_map)
            static_transformStamped.transform.rotation = Quaternion(*quat)

        self.broadcaster.sendTransform(static_transformStamped)

    def crop_curr_scan(self):
        scan = np.array(self.cur_scan.points)
        scan = np.column_stack([scan, np.ones(len(scan))])

        T_odom_to_base_link = self.pose_to_mat(self.cur_odom.pose)
        ref_point = T_odom_to_base_link[:3, 3]

        phase_name = self.phase_list[self.phase]
        scan_range = self.phase_parameters[phase_name]["scan_range"]
        indices = np.where(self.dist_square(scan[:, 0], scan[:, 1], scan[:, 2], ref_point) < scan_range * scan_range)

        crop_scan = o3d.geometry.PointCloud()
        crop_scan.points = o3d.utility.Vector3dVector(np.squeeze(scan[indices, :3]))

        return crop_scan

    def crop_global_map(self):

        T_odom_to_base_link = self.pose_to_mat(self.cur_odom.pose)
        T_map_to_base_link = np.matmul(self.T_map_to_odom, T_odom_to_base_link)
        ref_point = T_map_to_base_link[:3, 3]

        phase_name = self.phase_list[self.phase]

        global_map = np.array(self.global_maps[phase_name].points)
        global_map = np.column_stack([global_map, np.ones(len(global_map))])

        map_range = self.phase_parameters[phase_name]["map_range"]
        indices = np.where(
            self.dist_square(global_map[:, 0], global_map[:, 1], global_map[:, 2], ref_point) < map_range * map_range)

        crop_map = o3d.geometry.PointCloud()
        crop_map.points = o3d.utility.Vector3dVector(np.squeeze(global_map[indices, :3]))

        # publish map point cloud
        header = self.cur_odom.header
        header.frame_id = 'map'
        self.publish_point_cloud(self.pub_submap, header, np.array(crop_map.points))

        return crop_map

    def registration_at_scale(self, pc_scan, pc_map, initial, max_iteration = 100):


        phase_name = self.phase_list[self.phase]
        scan_voxel_size = self.phase_parameters[phase_name]["scan_voxel_size"]
        map_voxel_size = self.phase_parameters[phase_name]["map_voxel_size"]
        max_correst_dist = self.phase_parameters[phase_name]["max_corres_dist"]

        result_icp = o3d_registration.registration_icp(
            self.voxel_down_sample(pc_scan, scan_voxel_size),
            self.voxel_down_sample(pc_map, map_voxel_size),
            max_correst_dist, initial,
            o3d_registration.TransformationEstimationPointToPoint(),
            o3d_registration.ICPConvergenceCriteria(max_iteration=max_iteration))
        return result_icp.transformation, result_icp.fitness

    def cb_cur_odom(self, odom_msg):

        self.cur_odom = odom_msg

    def cb_cur_scan(self, pc_msg):

        # pc_msg.header.frame_id = 'camera_init'
        # pc_msg.header.stamp = rospy.Time().now()

        pc_msg.fields = [pc_msg.fields[0], pc_msg.fields[1], pc_msg.fields[2],
                         pc_msg.fields[4], pc_msg.fields[5], pc_msg.fields[6],
                         pc_msg.fields[3], pc_msg.fields[7]]
        pc = self.msg_to_array(pc_msg)

        self.stack_cnt += 1
        if self.stack_scan is None:
            self.stack_scan = np.squeeze(pc[:, :3])
        else:
            self.stack_scan = np.concatenate([self.stack_scan, np.squeeze(pc[:, :3])])

        # print(stack_scan.shape)
        if self.stack_cnt == self.scan_stack_size:
            self.receive_new_scan = True
            self.stack_cnt = 0
            self.cur_scan = o3d.geometry.PointCloud()
            self.cur_scan.points = o3d.utility.Vector3dVector(self.stack_scan)
            self.stack_scan = None

            # TODO: publish the cropped scan data
            self.publish_point_cloud(self.pub_pc_in_map, pc_msg.header, np.array(self.cur_scan.points))


    def voxel_down_sample(self, pcd, voxel_size):
        try:
            pcd_down = pcd.voxel_down_sample(voxel_size)
        except:
            # for opend3d 0.7 or lower
            pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
        return pcd_down

    def publish_point_cloud(self, publisher, header, pc):

        if self.headless:
            return

        data = np.zeros(len(pc), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
        ])
        data['x'] = pc[:, 0]
        data['y'] = pc[:, 1]
        data['z'] = pc[:, 2]
        if pc.shape[1] == 4:
            data['intensity'] = pc[:, 3]
        msg = ros_numpy.msgify(PointCloud2, data)
        msg.header = header
        publisher.publish(msg)

    def dist_square(self, x, y, z, ref = [0, 0, 0]):
        return (x - ref[0]) * (x - ref[0]) + (y - ref[1]) * (y - ref[1]) + (z - ref[2]) * (z - ref[2])

    def pose_to_mat(self, pose_msg):
        return np.matmul(
            tf.listener.xyz_to_mat44(pose_msg.pose.position),
            tf.listener.xyzw_to_mat44(pose_msg.pose.orientation))

    def msg_to_array(self, pc_msg):
        pc_array = ros_numpy.numpify(pc_msg)
        pc = np.zeros([len(pc_array), 3])
        pc[:, 0] = pc_array['x']
        pc[:, 1] = pc_array['y']
        pc[:, 2] = pc_array['z']
        return pc

    def inverse_se3(self, trans):
        trans_inverse = np.eye(4)
        # R
        trans_inverse[:3, :3] = trans[:3, :3].T
        # t
        trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
        return trans_inverse


if __name__ == '__main__':

    rospy.init_node('fast_lio_localization')
    localization_node = GlobalLocalization()

    rospy.spin()
