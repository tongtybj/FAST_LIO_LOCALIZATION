#!/usr/bin/env python
# coding=utf8
from __future__ import print_function, division, absolute_import

import copy
import time

import open3d as o3d
import rospy
import ros_numpy
from geometry_msgs.msg import PoseStamped, Pose, Point, Vector3, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import numpy as np
import tf
import tf.transformations
import tf2_ros


global_map = None
initialized = False
T_map_to_odom = np.eye(4)
cur_odom = None
cur_scan = None
new_scan = False
headless = False

np.set_printoptions(precision=3, suppress=True)

def dist_square(x, y, z):
    return x * x + y * y + z * z


def pose_to_mat(pose_msg):
    return np.matmul(
        tf.listener.xyz_to_mat44(pose_msg.pose.position),
        tf.listener.xyzw_to_mat44(pose_msg.pose.orientation))

def msg_to_array(pc_msg):
    pc_array = ros_numpy.numpify(pc_msg)
    pc = np.zeros([len(pc_array), 3])
    pc[:, 0] = pc_array['x']
    pc[:, 1] = pc_array['y']
    pc[:, 2] = pc_array['z']
    return pc

def registration_at_scale(pc_scan, pc_map, initial, scale):
    result_icp = o3d.pipelines.registration.registration_icp(
        voxel_down_sample(pc_scan, SCAN_VOXEL_SIZE * scale), voxel_down_sample(pc_map, MAP_VOXEL_SIZE * scale),
        1.0 * scale, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
    )
    return result_icp.transformation, result_icp.fitness

def inverse_se3(trans):
    trans_inverse = np.eye(4)
    # R
    trans_inverse[:3, :3] = trans[:3, :3].T
    # t
    trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
    return trans_inverse

def publish_point_cloud(publisher, header, pc):

    global headless
    if headless:
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


def crop_global_map_in_FOV(global_map, pose_estimation, cur_odom):

    T_odom_to_base_link = pose_to_mat(cur_odom.pose)
    T_map_to_base_link = np.matmul(pose_estimation, T_odom_to_base_link)
    T_base_link_to_map = inverse_se3(T_map_to_base_link)

    # convert map pointcloud w.r.t. LiDAR frame
    global_map_in_map = np.array(global_map.points)
    global_map_in_map = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
    global_map_in_base_link = np.matmul(T_base_link_to_map, global_map_in_map.T).T

    # extract pointcloud inside the FoV and range
    if FOV == 2 * np.pi:
        indices = np.where(
            dist_square(global_map_in_base_link[:, 0], global_map_in_base_link[:, 1], global_map_in_base_link[:, 2]) < FOV_FAR * FOV_FAR)
    elif FOV > np.pi:
        # All-range LiDAR
        indices = np.where(
            (dist_square(global_map_in_base_link[:, 0], global_map_in_base_link[:, 1], global_map_in_base_link[:, 2]) < FOV_FAR * FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    else:
        # Front view type LiDAR
        indices = np.where(
            (global_map_in_base_link[:, 0] > 0) &
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    global_map_in_FOV = o3d.geometry.PointCloud()
    global_map_in_FOV.points = o3d.utility.Vector3dVector(np.squeeze(global_map_in_map[indices, :3]))


    # publish map point cloud inside the FoV
    header = cur_odom.header
    header.frame_id = 'map'
    publish_point_cloud(pub_submap, header, np.array(global_map_in_FOV.points)[::10])

    return global_map_in_FOV


def global_localization(pose_estimation):
    global global_map, cur_scan, cur_odom, T_map_to_odom, new_scan


    if not new_scan:
        return False

    rospy.loginfo('Global localization by scan-to-map matching......')

    scan_tobe_mapped = copy.copy(cur_scan)

    tic = time.time()

    global_map_in_FOV = crop_global_map_in_FOV(global_map, pose_estimation, cur_odom)

    # rough ICP point matching
    transformation, _ = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5)

    # accurate ICP point matching
    transformation, fitness = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=transformation,
                                                    scale=1)
    toc = time.time()
    rospy.loginfo('Time: {:.2f}; fitness score:{:.2f}'.format(toc - tic, fitness))

    new_scan = False

    if fitness > LOCALIZATION_TH:
        T_map_to_odom = transformation

        # publish map_to_odom and tf
        map_to_odom = Odometry()
        xyz = tf.transformations.translation_from_matrix(T_map_to_odom)
        quat = tf.transformations.quaternion_from_matrix(T_map_to_odom)
        euler = np.array(tf.transformations.euler_from_quaternion(quat))
        map_to_odom.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        map_to_odom.header.stamp = cur_odom.header.stamp
        map_to_odom.header.frame_id = 'map'
        pub_map_to_odom.publish(map_to_odom)

        static_transformStamped = TransformStamped()

        static_transformStamped.header.stamp = cur_odom.header.stamp
        static_transformStamped.header.frame_id = 'map'
        static_transformStamped.child_frame_id = 'camera_init'

        pos = tf.transformations.translation_from_matrix(T_map_to_odom)
        static_transformStamped.transform.translation = Vector3(*pos)
        quat = tf.transformations.quaternion_from_matrix(T_map_to_odom)
        static_transformStamped.transform.rotation = Quaternion(*quat)

        broadcaster.sendTransform(static_transformStamped)

        # br.sendTransform(tf.transformations.translation_from_matrix(T_map_to_odom),
        #                  tf.transformations.quaternion_from_matrix(T_map_to_odom),
        #                  cur_odom.header.stamp,
        #                  'camera_init', 'map')

        rospy.loginfo("Map to Odom: pos: {}, euler: {} \n".format(xyz, euler))
        return True
    else:
        rospy.logwarn('Not match!!!! \n')
        return False



def voxel_down_sample(pcd, voxel_size):
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    except:
        # for opend3d 0.7 or lower
        pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    return pcd_down


def initialize_global_map(pc_msg):
    global global_map

    global_map = o3d.geometry.PointCloud()
    global_map.points = o3d.utility.Vector3dVector(msg_to_array(pc_msg)[:, :3])
    global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)
    rospy.loginfo('Global map received.')


def cb_save_cur_odom(odom_msg):
    global cur_odom
    cur_odom = odom_msg


def cb_save_cur_scan(pc_msg):
    global cur_scan
    global new_scan
    global headless
    pc_msg.header.frame_id = 'camera_init'
    pc_msg.header.stamp = rospy.Time().now()

    pc_msg.fields = [pc_msg.fields[0], pc_msg.fields[1], pc_msg.fields[2],
                     pc_msg.fields[4], pc_msg.fields[5], pc_msg.fields[6],
                     pc_msg.fields[3], pc_msg.fields[7]]
    pc = msg_to_array(pc_msg)

    # TODO: shift to "global_localization" to save computation
    indices = np.where(dist_square(pc[:, 0], pc[:, 1], pc[:, 2]) < SCAN_THRESH * SCAN_THRESH)

    cur_scan = o3d.geometry.PointCloud()
    cur_scan.points = o3d.utility.Vector3dVector(np.squeeze(pc[indices, :3]))

    new_scan = True

    if not headless:
        publish_point_cloud(pub_pc_in_map, pc_msg.header, np.array(cur_scan.points)[::10])
        # pub_pc_in_map.publish(pc_msg)


def thread_localization(msg):

    global T_map_to_odom
    global_localization(T_map_to_odom)



if __name__ == '__main__':

    rospy.init_node('fast_lio_localization')
    rospy.loginfo('Localization Node Inited...')

    # parameter for registration
    MAP_VOXEL_SIZE = rospy.get_param("~registration/map_voxel_size", 0.4)
    SCAN_VOXEL_SIZE = rospy.get_param("~registration/scan_voxel_size", 0.1)

    # Global localization frequency (HZ)
    FREQ_LOCALIZATION = rospy.get_param("~registration/freq_localization", 0.5)

    # The threshold of global localization,
    # only those scan2map-matching with higher fitness than LOCALIZATION_TH will be taken
    LOCALIZATION_TH = rospy.get_param("~registration/localization_thresh", 0.95)

    # FOV(rad), modify this according to your LiDAR type
    FOV = rospy.get_param("~registration/fov", 2 * np.pi)

    # The farthest distance(meters) within FOV
    FOV_FAR = rospy.get_param("~registration/fov_far", 10.0)

    # The range of current scan
    SCAN_THRESH = rospy.get_param("~registration/scan_thresh", 5.0)

    # publisher
    pub_pc_in_map = rospy.Publisher('/cur_scan_in_map', PointCloud2, queue_size=1)
    pub_submap = rospy.Publisher('/submap', PointCloud2, queue_size=1)
    pub_map_to_odom = rospy.Publisher('/map_to_odom', Odometry, queue_size=1)
    #br = tf.TransformBroadcaster()
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    rospy.Subscriber('/cloud_registered', PointCloud2, cb_save_cur_scan, queue_size=1)
    rospy.Subscriber('/Odometry', Odometry, cb_save_cur_odom, queue_size=1)

    headless = rospy.get_param("~headless", False)

    # get init frame offset from rosparam
    init_pos = Point()
    init_pos.x = rospy.get_param("~initial_frame_offset/x", 0)
    init_pos.y = rospy.get_param("~initial_frame_offset/y", 0)
    init_pos.z = rospy.get_param("~initial_frame_offset/z", 0)
    init_roll = rospy.get_param("~initial_frame_offset/roll", 0)
    init_pitch = rospy.get_param("~initial_frame_offset/pitch", 0)
    init_yaw = rospy.get_param("~initial_frame_offset/yaw", 0)
    init_q = Quaternion(*tf.transformations.quaternion_from_euler(init_roll, init_pitch, init_yaw))
    initial_pose =  np.matmul(tf.listener.xyz_to_mat44(init_pos),
                              tf.listener.xyzw_to_mat44(init_q))

    rospy.logwarn('Waiting for global map......')
    initialize_global_map(rospy.wait_for_message('/3d_map', PointCloud2))
    rospy.loginfo('Get global map......')

    while not initialized:

        if new_scan:
            initialized = global_localization(initial_pose)
        else:
            rospy.logwarn('Waiting for current scan data')

        rospy.sleep(1.0)

    rospy.loginfo('Initialize successfully!!!!!! \n')

    rospy.Timer(rospy.Duration(1 / FREQ_LOCALIZATION), thread_localization)

    rospy.spin()
