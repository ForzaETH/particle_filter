import rospy
import numpy as np
import tf2_ros
import tf.transformations as tft
import utils.utils as Utils
import range_libc
from threading import Lock

# Message types
from geometry_msgs.msg import PoseStamped, PoseArray, PointStamped, \
    PoseWithCovarianceStamped, TransformStamped, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, MapMetaData
from nav_msgs.srv import GetMap

# Define Datatypes
from enum import Enum
from typing import Optional

# Debug
from time import time
from collections import deque
import cProfile
import pstats

# Dynamic Reconfigure
from dynamic_reconfigure.msg import Config


class RangeLibVariant(Enum):
    '''
    These flags indicate several variants of the sensor model. Only one of them is used at a time.
    '''
    VAR_NO_EVAL_SENSOR_MODEL = 0
    VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
    VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
    VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
    VAR_RADIAL_CDDT_OPTIMIZATIONS = 4


class ParticleFilter():
    '''
    Particle Filter Two: Electric Boogaloo.

    A refactor of the MIT Racecar Particle Filter code with augmentations from other places,
    notably the TUM paper "ROS-based localization of a race vehicle at high-speed using LIDAR".
    '''

    def __init__(self):
        # parameters
        self.MAX_PARTICLES = int(rospy.get_param("~max_particles"))
        self.MAX_VIZ_PARTICLES = int(rospy.get_param("~max_viz_particles"))
        self.INV_SQUASH_FACTOR = 1.0 / float(rospy.get_param("~squash_factor"))
        self.MAX_RANGE_METERS = float(rospy.get_param("~max_range"))
        self.WHICH_RM = rospy.get_param("~range_method", "cddt").lower()
        self.RANGELIB_VAR = RangeLibVariant(
            rospy.get_param("~rangelib_variant", "3"))
        self.POSE_PUB_TOPIC = str(rospy.get_param(
            "~pose_pub_topic", "/tracked_pose"))
        self.DO_VIZ = bool(rospy.get_param("~viz"))
        self.PUBLISH_TF = bool(rospy.get_param("~publish_tf", True))
        self.ODOM_TOPIC = rospy.get_param("~odometry_topic")
        self.THETA_DISCRETIZATION = int(
            rospy.get_param("~theta_discretization", 112))
        '''Number of discrete bins for angular values for the (P)CDDT and LUT methods'''
        self.PUB_COVARIANCE = bool(
            rospy.get_param("~pub_covariance", False))
        '''Whether or not to publish an empirically-calculated covariance'''

        # (Re) initialization constants
        self.INIT_VAR_X = float(rospy.get_param("~initial_var_x", 0.5))
        self.INIT_VAR_Y = float(rospy.get_param("~initial_var_y", 0.5))
        self.INIT_VAR_TH = float(rospy.get_param("~initial_var_theta", 0.4))

        # sensor model constants
        self.Z_HIT = float(rospy.get_param("~z_hit", 0.75))
        self.Z_SHORT = float(rospy.get_param("~z_short", 0.01))
        self.Z_MAX = float(rospy.get_param("~z_max", 0.07))
        self.Z_RAND = float(rospy.get_param("~z_rand", 0.12))
        self.SIGMA_HIT = float(rospy.get_param("~sigma_hit", 0.4))
        self.LAM_SHORT = float(rospy.get_param("~lambda_short", 0.1))

        # motion model constants
        self.MOTION_MODEL = rospy.get_param("~motion_model", "tum").lower()
        if self.MOTION_MODEL == 'tum':
            self.ALPHA_1 = float(rospy.get_param("~alpha_1_tum"))
            self.ALPHA_2 = float(rospy.get_param("~alpha_2_tum"))
            self.ALPHA_3 = float(rospy.get_param("~alpha_3_tum"))
            self.ALPHA_4 = float(rospy.get_param("~alpha_4_tum"))

            rospy.loginfo("PF2 initial parameters...")
            rospy.loginfo(f"PF2: alpha1: {self.ALPHA_1}")
            rospy.loginfo(f"PF2: alpha2: {self.ALPHA_2}")
            rospy.loginfo(f"PF2: alpha3: {self.ALPHA_3}")
            rospy.loginfo(f"PF2: alpha4: {self.ALPHA_4}")

            self.LAM_THRESH = float(rospy.get_param("~lam_thresh"))
            rospy.loginfo(f"PF2: lam_thresh: {self.LAM_THRESH}")
        elif self.MOTION_MODEL == 'amcl':
            self.ALPHA_1 = float(rospy.get_param("~alpha_1_amcl"))
            self.ALPHA_2 = float(rospy.get_param("~alpha_2_amcl"))
            self.ALPHA_3 = float(rospy.get_param("~alpha_3_amcl"))
            self.ALPHA_4 = float(rospy.get_param("~alpha_4_amcl"))
        elif self.MOTION_MODEL == 'arc':
            rospy.logwarn("Using arc motion model - may not be fully tested.")
            self.MOTION_DISPERSION_ARC_X = float(
                rospy.get_param("~motion_dispersion_arc_x", 0.05))
            self.MOTION_DISPERSION_ARC_Y = float(
                rospy.get_param("~motion_dispersion_arc_y", 0.025))
            self.MOTION_DISPERSION_ARC_THETA = float(
                rospy.get_param("~motion_dispersion_arc_theta", 0.25))
            self.MOTION_DISPERSION_ARC_XY = float(
                rospy.get_param("~motion_dispersion_arc_xy", 0))
            # self.MOTION_DISPERSION_ARC_XY_MAX = float(
            #     rospy.get_param("~motion_dispersion_arc_xy_max", 0))
            # self.MOTION_DISPERSION_ARC_XTHETA = float(
            #     rospy.get_param("~motion_dispersion_arc_xtheta", 0))
            self.MOTION_DISPERSION_ARC_X_MIN = float(
                rospy.get_param("~motion_dispersion_arc_x_min", 0.01))
            self.MOTION_DISPERSION_ARC_Y_MIN = float(
                rospy.get_param("~motion_dispersion_arc_y_min", 0.01))
            self.MOTION_DISPERSION_ARC_Y_MAX = float(
                rospy.get_param("~motion_dispersion_arc_y_max", 0.01))
            self.MOTION_DISPERSION_ARC_THETA_MIN = float(
                rospy.get_param("~motion_dispersion_arc_theta_min", 0.01))
            self.MOTION_DISPERSION_ARC_XY_MIN_X = float(
                rospy.get_param("~motion_dispersion_arc_xy_min_x", 0.01))
        else:
            rospy.logwarn("Using default MIT PF motion model - may not be fully tested.")
            self.MOTION_DISPERSION_X = float(
                rospy.get_param("~motion_dispersion_x", 0.05))
            self.MOTION_DISPERSION_Y = float(
                rospy.get_param("~motion_dispersion_y", 0.025))
            self.MOTION_DISPERSION_THETA = float(
                rospy.get_param("~motion_dispersion_theta", 0.25))

        # Boxed lidar model paramters.
        # The defaults are based on the Hokyuo laser scan.
        self.LIDAR_ASPECT_RATIO = float(
            rospy.get_param("~lidar_aspect_ratio", 3.0))
        self.DES_LIDAR_BEAMS = int(rospy.get_param("~des_lidar_beams", 21))
        '''Desired number of beams. Will be an odd number'''

        scan_msg : LaserScan = rospy.wait_for_message('/scan', LaserScan, None)
        self.NUM_LIDAR_BEAMS = len(scan_msg.ranges)
        self.START_THETA = scan_msg.angle_min
        self.END_THETA = scan_msg.angle_max

        # Data members in the Particle Filter
        self.state_lock = Lock()
        '''Lock to prevent multithreading errors'''
        self.rate = rospy.Rate(200)
        '''Enforces update rate. This should not be lower than the odom message rate (50)'''
        self.MAX_RANGE_PX: int = None
        '''Maximum lidar range in pixels'''
        self.odometry_data = np.array([0.0, 0.0, 0.0])
        '''NDArray Buffer for odometry data (x, y, theta) representing the change in last and current odom position'''
        self.map_info: MapMetaData = None
        '''Buffer for Occupancy Grid metadata'''
        self.permissible_region: np.ndarray = None
        '''NDArray containing the OccupancyGrid. 0: not permissible, 1: permissible.'''
        self.map_initialized = False
        '''Boolean flag set when `get_omap()` is called'''
        self.lidar_initialized = False
        '''Boolean flag set when the Lidar Scan arrays have been populated'''
        self.odom_initialized = False
        '''Boolean flag set when `self.odometry_data` has been initialized'''
        self.last_pose: np.ndarray = None
        '''3-vector holding the last-known pose from odometry (x,y,theta)'''
        self.curr_pose: np.ndarray = None
        '''3-vector holding the current pose from odometry (x,y,theta)'''
        self.range_method = None
        '''RangeLibc binding, set in `get_omap()`'''
        self.last_stamp: rospy.Time = rospy.Time(0)
        '''Timestamp of last-recieved odometry message'''
        self.last_pub_stamp = rospy.Time.now()
        '''Last published timestamp'''
        self.first_sensor_update = True
        '''Boolean flag for use in `sensor_model()`'''
        self.odom_msgs: deque = deque([], maxlen=5)
        '''Buffer holding the last few Odometry messages, which may be clumped for some reason.'''
        self.local_deltas = np.zeros((self.MAX_PARTICLES, 3))
        '''NDArray of local motion, allocated for use in motion model'''

        # cache these for the sensor model computation
        self.queries: np.ndarray = None
        '''NDArray of sensor queries (call to RangeLibc), init'd on the first `sensor_model()` call'''
        self.ranges: np.ndarray = None
        '''NDArray of ranges returned by RangeLibc'''
        self.tiled_angles: np.ndarray = None
        '''Used in `sensor_model()`'''
        self.sensor_model_table: np.ndarray = None
        '''NDArray containing precomputed, discretized sensor model probability init'd in `precompute_sensor_model()`'''
        self.LIDAR_SAMPLE_IDXS: np.ndarray = None
        '''NDArray holding the evenly spaced lidar indices to sample from, calcualted in `get_boxed_indices()`'''
        self.LIDAR_THETA_LUT: np.ndarray = None
        '''Single-precision NDArray lookup table of the lidar beam angles at the indices of LIDAR_SAMPLE_IDXS'''

        # particle poses and weights
        self.inferred_pose: np.ndarray = None
        '''NDArray of the expected value of the pose given the particle distribution'''
        self.particle_indices = np.arange(self.MAX_PARTICLES)
        '''Numbered list of particles.'''
        self.particles = np.zeros((self.MAX_PARTICLES, 3))
        '''NDArray of potential particles. Each represents a hypothesis location of the base link. (MAX_PARTICLES, 3)'''
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
        '''NDArray weighting the particles, initialized uniformly (MAX_PARTICLES, )'''
        if self.PUB_COVARIANCE:
            self.cov = np.zeros((3, 3))
            '''NDArray representing the covariance (x,y,theta)'''

        # Initialize Map and Sensor Model
        self.get_boxed_indices()
        self.get_omap()
        self.precompute_sensor_model()

        # Initialize position (if known). Else use "lost robot" mode
        if rospy.has_param("~initial_pose_x") and \
                rospy.has_param("~initial_pose_y") and \
                rospy.has_param("~initial_pose_theta"):

            START_X = rospy.get_param("~initial_pose_x")
            START_Y = rospy.get_param("~initial_pose_y")
            START_THETA = rospy.get_param("~initial_pose_theta")
            self.initialize_particles_pose(
                START_X, START_Y, posetheta=START_THETA)
        else:
            self.initialize_global()

        self.particle_pub = rospy.Publisher(
            "/pf/viz/particles", PoseArray, queue_size=1)
        '''Publishes particle cloud onto `/pf/viz/particles` (randomly sampled)'''

        self.pose_pub = rospy.Publisher(
            self.POSE_PUB_TOPIC, PoseStamped, queue_size=1)
        '''Publishes inferred pose on POSE_PUB_TOPIC (default: `/tracked_pose`)'''
        if self.PUB_COVARIANCE:
            self.pose_cov_pub = rospy.Publisher(
                self.POSE_PUB_TOPIC+'/with_covariance', PoseWithCovarianceStamped, queue_size=1)
            '''Publishes inferred pose with Covariance. (default: `/tracked_pose/with_covariance`)'''

        self.pub_tf = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Wait for transforms
        while not self.tf_buffer.can_transform("base_link", "laser", rospy.Time(), rospy.Duration(1.0)):
            rospy.logwarn("PF2 Waiting for base_link->laser transformation")
        rospy.loginfo("base_link->laser transformation OK")

        trans: TransformStamped = self.tf_buffer.lookup_transform(
            "base_link", "laser", rospy.Time())
        self.laser_base_link_offset = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            0.0])

        # ! Honestly this can just be a function, doesn't need to be a class
        self.particle_utils = Utils.ParticleUtils(self.laser_base_link_offset)

        rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"),
                         LaserScan, self.lidarCB, queue_size=2)
        # tcp_nodelay ensures we get a fixed 50hz, bypassing wacky buffering in the Transport Layer
        rospy.Subscriber(self.ODOM_TOPIC, Odometry, self.odomCB, queue_size=2, tcp_nodelay=True)
        # rospy.Subscriber(self.ODOM_TOPIC, Odometry, self.odomCB, queue_size=2, tcp_nodelay=False)
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                         self.clicked_poseCB, queue_size=1)
        rospy.Subscriber("/clicked_point", PointStamped,
                         self.clicked_poseCB, queue_size=1)
        rospy.Subscriber(
            "/dynamic_pf2_tuner_node/parameter_updates", Config, self.dyn_param_cb)

        rospy.loginfo("PF2 finished initializing.")

        self.DEBUG = bool(rospy.get_param("~debug", False))
        if self.DEBUG:
            self.sensor_model_time_ms = deque(maxlen=150)
            self.motion_model_time_ms = deque(maxlen=150)
            self.overall_time_ms = deque(maxlen=150)
            self.itr = 0

            self.profiler = cProfile.Profile()


        # Debugging stuff
        self.last_odom = rospy.Time.now().to_sec()
        self.last_odom_msg : Odometry = None
        '''The most-recently processed odom msg'''
        self.last_vesc = rospy.Time.now().to_sec()

    def get_omap(self):
        '''
        Fetch the occupancy grid map from the map_server instance, and initialize the correct
        RangeLibc method. Also stores a matrix which indicates the permissible region of the map
        '''
        rospy.wait_for_service("static_map")
        map_msg = rospy.ServiceProxy("static_map", GetMap)().map

        self.map_info = map_msg.info
        oMap = range_libc.PyOMap(map_msg)
        self.MAX_RANGE_PX = int(
            self.MAX_RANGE_METERS / self.map_info.resolution)

        # initialize range method
        rospy.loginfo(f"Initializing range method: {self.WHICH_RM}")
        if self.WHICH_RM == "bl":
            self.range_method = range_libc.PyBresenhamsLine(
                oMap, self.MAX_RANGE_PX)
        elif "cddt" in self.WHICH_RM:
            self.range_method = range_libc.PyCDDTCast(
                oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
            if self.WHICH_RM == "pcddt":
                rospy.loginfo("Pruning...")
                self.range_method.prune()
        elif self.WHICH_RM == "rm":
            self.range_method = range_libc.PyRayMarching(
                oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == "rmgpu":
            self.range_method = range_libc.PyRayMarchingGPU(
                oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == "glt":
            self.range_method = range_libc.PyGiantLUTCast(
                oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
        rospy.loginfo("Done loading map")

        # 0: permissible, -1: unmapped, 100: blocked
        array_255 = np.array(map_msg.data).reshape(
            (map_msg.info.height, map_msg.info.width))

        # 0: not permissible, 1: permissible
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255 == 0] = 1

        # // Sanity Check
        # _, axs = plt.subplots(nrows=2, ncols=1)
        # axs[0].set_title("Original data")
        # axs[0].imshow(array_255)
        # axs[1].set_title("Permissible Region")
        # im=axs[1].imshow(self.permissible_region)
        # plt.colorbar(im, orientation="horizontal")
        # plt.show()

        self.map_initialized = True

    def precompute_sensor_model(self):
        '''
        Generate and store a lookup table which represents the sensor model.

        For each discrete computed range value, this provides the probability of measuring that (discrete) range.

        This table is indexed by the sensor model at runtime by discretizing the measurements
        and computed ranges from RangeLibc.
        '''
        rospy.loginfo("Precomputing sensor model")
        # sensor model constants
        z_short = self.Z_SHORT
        z_max = self.Z_MAX
        z_rand = self.Z_RAND
        z_hit = self.Z_HIT
        # normalise sigma and lambda from meters to pixel space
        # [px] = [m] / [m/px]
        sigma_hit = self.SIGMA_HIT/self.map_info.resolution
        lam_short = self.LAM_SHORT/self.map_info.resolution

        table_width = int(self.MAX_RANGE_PX) + 1
        self.sensor_model_table = np.zeros((table_width, table_width))

        # compute normalizers for the gaussian and exponential distributions
        norm_gau = np.zeros((table_width,))
        norm_exp = np.zeros((table_width,))
        for d in range(table_width):
            sum_gau = 0
            sum_exp = 0
            for r in range(table_width):
                z = float(d-r)
                sum_gau += np.exp(-(z*z)/(2.0*sigma_hit*sigma_hit)) / \
                        (sigma_hit * np.sqrt(2.0*np.pi))

                if r <= d:
                    sum_exp += ( lam_short * np.exp(-lam_short*r) )

            norm_gau[d] = 1/sum_gau
            norm_exp[d] = 1/sum_exp

        # d is the computed range from RangeLibc (predicted range)
        for d in range(table_width):
            norm = 0.0
            # r is the observed range from the lidar unit
            for r in range(table_width):
                prob = 0.0
                z = float(d-r)

                # Probability of hitting the intended object
                # P_hit -- sample from a Gaussian
                prob += z_hit * \
                    ( np.exp(-(z*z)/(2.0*sigma_hit*sigma_hit)) / \
                    (sigma_hit * np.sqrt(2.0*np.pi)) ) * norm_gau[d]

                # observed range is less than the predicted range - short reading
                # P_short -- sample from exponential distribution
                # note: z must be positive here!
                if r <= d:
                    prob += z_short * ( lam_short * np.exp(-lam_short*r) ) * norm_exp[d]

                # erroneous max range measurement
                # P_max -- uniform distribution at max range
                if r == int(self.MAX_RANGE_PX):
                    prob += z_max

                # random measurement
                # P_rand -- uniform distribution across entire range
                if r < self.MAX_RANGE_PX:
                    prob += z_rand * 1.0/self.MAX_RANGE_PX

                norm += prob
                self.sensor_model_table[r, d] = prob

            # normalize
            self.sensor_model_table[:, d] /= norm

        # upload the sensor model to RangeLib for acceleration
        if self.RANGELIB_VAR.value > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    def get_boxed_indices(self):
        '''
        Finds an evenly spaced "boxed" pattern of beams based on the TUM paper
        "ROS-based localization of a race vehicle at high-speed using LIDAR".
        '''
        beam_angles = np.linspace(
            self.START_THETA, self.END_THETA, self.NUM_LIDAR_BEAMS)

        MID_IDX = self.NUM_LIDAR_BEAMS//2
        sparse_idxs = [MID_IDX]

        # Structures
        a = self.LIDAR_ASPECT_RATIO
        beam_proj = 2*a*np.array([np.cos(beam_angles), np.sin(beam_angles)])
        # Allows us to do intersection math later
        beam_intersections = np.zeros((2, self.NUM_LIDAR_BEAMS))

        # Compute the points of intersection along a uniform corridor of given aspect ratio
        box_corners = [(a, 1), (a, -1), (-a, -1), (-a, 1)]
        for idx in range(len(box_corners)):
            x1, y1 = box_corners[idx]
            x2, y2 = box_corners[0] if idx == 3 else box_corners[idx+1]
            for i in range(self.NUM_LIDAR_BEAMS):
                x4 = beam_proj[0, i]
                y4 = beam_proj[1, i]

                den = (x1-x2)*(-y4)-(y1-y2)*(-x4)
                if den == 0:
                    continue    # parallel lines

                t = ((x1)*(-y4)-(y1)*(-x4))/den
                u = ((x1)*(y1-y2)-(y1)*(x1-x2))/den

                px = u*x4
                py = u*y4
                if 0 <= t <= 1.0 and 0 <= u <= 1.0:
                    beam_intersections[0, i] = px
                    beam_intersections[1, i] = py

        # Compute the distances for uniform spacing
        dx = np.diff(beam_intersections[0, :])
        dy = np.diff(beam_intersections[1, :])
        dist = np.sqrt(dx**2 + dy**2)
        total_dist = np.sum(dist)
        dist_amt = total_dist/(self.DES_LIDAR_BEAMS-1)
        # rospy.loginfo(f"{dist.shape=}, {total_dist=:.2f}, {dist_amt=:.2f}")

        # Calc half of the evenly-spaced interval first, then the other half
        idx = MID_IDX + 1
        DES_BEAMS2 = self.DES_LIDAR_BEAMS//2 + 1
        acc = 0
        while len(sparse_idxs) <= DES_BEAMS2:
            acc += dist[idx]
            if acc >= dist_amt:
                acc = 0
                sparse_idxs.append(idx-1)
            idx += 1

            if idx == self.NUM_LIDAR_BEAMS-1:
                sparse_idxs.append(self.NUM_LIDAR_BEAMS-1)
                break

        mirrored_half = []
        for idx in sparse_idxs[1:]:
            new_idx = 2*sparse_idxs[0]-idx
            mirrored_half.insert(0, new_idx)
        sparse_idxs = mirrored_half + sparse_idxs

        self.LIDAR_SAMPLE_IDXS = np.array(sparse_idxs)
        self.LIDAR_THETA_LUT = beam_angles[self.LIDAR_SAMPLE_IDXS]
        self.LIDAR_THETA_LUT = self.LIDAR_THETA_LUT.astype(np.single)

    def initialize_global(self):
        '''
        Spread the particle distribution over the permissible region of the state space.

        Future Extension: Informed sampling by spreading over the race line
        '''
        while self.state_lock.locked():
            rospy.loginfo_once("PF2 Global Initialization: Waiting for state to become unlocked")
            rospy.sleep(0.1)

        self.state_lock.acquire()
        rospy.loginfo("Lost Robot Initialization")

        # randomize over grid coordinate space
        permissible_x, permissible_y = np.where(self.permissible_region == 1)
        indices = np.random.randint(
            0, len(permissible_x), size=self.MAX_PARTICLES)

        permissible_states = np.zeros((self.MAX_PARTICLES, 3))
        permissible_states[:, 0] = permissible_y[indices]
        permissible_states[:, 1] = permissible_x[indices]
        permissible_states[:, 2] = np.random.random(
            self.MAX_PARTICLES) * np.pi * 2.0

        Utils.map_to_world(permissible_states, self.map_info)
        self.particles = permissible_states
        self.weights[:] = 1.0 / self.MAX_PARTICLES

        self.state_lock.release()

    def initialize_particles_pose(self, posex: float, posey: float,
                                  posetheta: Optional[float] = None,
                                  poseo: Optional[Quaternion] = None):
        '''
        Initialize particles in the general region of the provided pose.

        Either initialize with a yaw (theta) or Quaternion.
        '''
        assert not (posetheta is None and poseo is None)
        assert not (posetheta is not None and poseo is not None)

        init_var_x = self.INIT_VAR_X
        init_var_y = self.INIT_VAR_Y
        init_var_th = self.INIT_VAR_TH

        while self.state_lock.locked():
            rospy.loginfo_once("PF2 Pose Initialization: Waiting for state to become unlocked")
            rospy.sleep(0.1)

        self.state_lock.acquire()

        if poseo is not None:
            posetheta = Utils.quaternion_to_angle(poseo)

        rospy.loginfo(
            f"Setting initial pose at x:{posex:.2f}, y:{posey:.2f}, theta:{np.degrees(posetheta):.2f}deg")
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
        self.particles[:, 0] = posex + \
            np.random.normal(scale=init_var_x, size=self.MAX_PARTICLES)
        self.particles[:, 1] = posey + \
            np.random.normal(scale=init_var_y, size=self.MAX_PARTICLES)
        self.particles[:, 2] = posetheta + \
            np.random.normal(scale=init_var_th, size=self.MAX_PARTICLES)

        self.state_lock.release()

    # Callbacks
    def dyn_param_cb(self, _):
        '''Reads relevant dynamically reconfigurable params off the parameter server'''
        if self.MOTION_MODEL == 'tum' or self.MOTION_MODEL == 'amcl':
            self.ALPHA_1 = float(rospy.get_param("dynamic_pf2_tuner_node/alpha_1"))
            self.ALPHA_2 = float(rospy.get_param("dynamic_pf2_tuner_node/alpha_2"))
            self.ALPHA_3 = float(rospy.get_param("dynamic_pf2_tuner_node/alpha_3"))
            self.ALPHA_4 = float(rospy.get_param("dynamic_pf2_tuner_node/alpha_4"))

            rospy.loginfo("PF2 dynamic reconfigure...")
            rospy.loginfo(f"PF2: alpha1: {self.ALPHA_1}")
            rospy.loginfo(f"PF2: alpha2: {self.ALPHA_2}")
            rospy.loginfo(f"PF2: alpha3: {self.ALPHA_3}")
            rospy.loginfo(f"PF2: alpha4: {self.ALPHA_4}")

            if self.MOTION_MODEL == 'tum':
                self.LAM_THRESH = float(rospy.get_param("dynamic_pf2_tuner_node/lam_thresh"))
                rospy.loginfo(f"PF2: lam_thresh: {self.LAM_THRESH}")

    def lidarCB(self, msg: LaserScan):
        self.downsampled_ranges = np.array(msg.ranges)[self.LIDAR_SAMPLE_IDXS]
        self.lidar_initialized = True

    def odomCB(self, msg: Odometry):
        self.odom_msgs.append(msg)
        self.last_odom_msg = msg
        # rospy.loginfo(f"Odom gap: {(rospy.Time.now().to_sec()-self.last_odom)*1000}ms")
        # self.last_odom = rospy.Time.now().to_sec()

        # if self.last_odom_msg is not None:
        #     dx = self.last_odom_msg.pose.pose.position.x - msg.pose.pose.position.x
        #     dy = self.last_odom_msg.pose.pose.position.y - msg.pose.pose.position.y
        #     rospy.loginfo(f"{dx=:.6f}, {dy=:.6f}")

    def clicked_poseCB(self, msg):
        '''
        Receive pose messages from RViz and initialize the particle distribution in response.
        '''
        if isinstance(msg, PointStamped):
            rospy.loginfo(
                "Recieved PointStamped message, re-initializing globally")
            self.initialize_global()
        elif isinstance(msg, PoseWithCovarianceStamped):
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            o = msg.pose.pose.orientation
            self.initialize_particles_pose(x, y, poseo=o)

    # Visualize and Publish

    def publish_tf(self, pose, stamp):
        """ Publish tf and Pose messages for the car. """

        # Avoid re-publishing stamp
        if stamp.to_sec() <= self.last_pub_stamp.to_sec():
            return

        map_base_link_pos = pose[0:2]
        map_laser_rotation = tft.quaternion_from_euler(0, 0, pose[2])

        header = Utils.make_header("map", stamp)

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = map_base_link_pos[0]
        pose_msg.pose.position.y = map_base_link_pos[1]
        pose_msg.pose.orientation.x = map_laser_rotation[0]
        pose_msg.pose.orientation.y = map_laser_rotation[1]
        pose_msg.pose.orientation.z = map_laser_rotation[2]
        pose_msg.pose.orientation.w = map_laser_rotation[3]
        self.pose_pub.publish(pose_msg)

        if self.PUB_COVARIANCE:
            pose_cov_msg = PoseWithCovarianceStamped()
            pose_cov_msg.pose.pose = pose_msg.pose
            pose_cov_msg.header = header

            covariance = np.zeros((6, 6))
            covariance[0:2, 0:2] = self.cov[0:2, 0:2]   # xy covariances
            covariance[5, 5] = self.cov[2, 2]           # theta variance
            covariance[0:2, 5] = self.cov[0:2, 2]       # xy-theta covariance
            covariance[5, 0:2] = self.cov[0:2, 2]
            pose_cov_msg.pose.covariance = covariance.flatten().tolist()
            self.pose_cov_pub.publish(pose_cov_msg)

        # rospy.loginfo(f"Inferred pose at x: {pose[0]:.2f}, y: {pose[1]:.2f}")

        if self.PUBLISH_TF:
            t = TransformStamped()
            t.header = header
            t.child_frame_id = "base_link"
            t.transform.translation.x = map_base_link_pos[0]
            t.transform.translation.y = map_base_link_pos[1]
            t.transform.translation.z = 0.0
            t.transform.rotation.x = map_laser_rotation[0]
            t.transform.rotation.y = map_laser_rotation[1]
            t.transform.rotation.z = map_laser_rotation[2]
            t.transform.rotation.w = map_laser_rotation[3]

            # Publish position/orientation for car_state
            self.pub_tf.sendTransform(t)

        self.last_pub_stamp = self.last_stamp

    def visualise(self):
        '''
        Publish visualization of the particles
        '''
        if not self.DO_VIZ:
            return

        def publish_particles(particles):
            # publish the given particles as a PoseArray object
            pa = PoseArray()
            pa.header = Utils.make_header("map")
            pa.poses = self.particle_utils.particles_to_poses(particles)
            self.particle_pub.publish(pa)

        if self.particle_pub.get_num_connections() > 0:
            # publish a downsampled version of the particle distribution to avoid latency
            if self.MAX_PARTICLES > self.MAX_VIZ_PARTICLES:
                proposal_indices = np.random.choice(
                    self.particle_indices, self.MAX_VIZ_PARTICLES, p=self.weights)
                publish_particles(self.particles[proposal_indices, :])
            else:
                publish_particles(self.particles)

    # AMCL Part
    def motion_model(self, proposal_dist, action):
        '''
        The motion model applies the odometry to the particle distribution.

        proposal_dist is a numpy array representing the current belief of the base link.

        action represents dx,dy,dtheta in the lidar frame.

        Uses the MOTION_MODEL parameter (amcl | tum | default) to apply the required transforms.
        '''
        if self.DEBUG:
            tic = time()

        if self.MOTION_MODEL == 'amcl' or self.MOTION_MODEL == 'tum':
            # Taken from AMCL ROS Package
            a1 = self.ALPHA_1
            a2 = self.ALPHA_2
            a3 = self.ALPHA_3
            a4 = self.ALPHA_4

            dx = self.curr_pose[0]-self.last_pose[0]
            dy = self.curr_pose[1]-self.last_pose[1]
            dtheta = Utils.angle_diff(self.curr_pose[2], self.last_pose[2])
            d_trans = np.sqrt(dx**2 + dy**2)

            # To prevent drift/instability when we are not moving
            if d_trans < 0.01:
                return

            # Dont calculate rot1 if we are "rotating in place"
            d_rot1 = Utils.angle_diff(np.arctan2(dy, dx), self.last_pose[2]) \
                if d_trans > 0.01 else 0.0

            reverse_offset = 0.0
            reverse_spread = 1.0

            # check if we are reversing
            if len(self.odom_msgs) and self.odom_msgs[-1].twist.twist.linear.x < -0.05:
                reverse_offset = np.pi
                reverse_spread = 1.1
                # Rotate d_rot1 180 degrees
                d_rot1 += np.pi if d_rot1 < -np.pi/2 else -np.pi

            d_rot2 = Utils.angle_diff(dtheta, d_rot1)

            # Enables this to happen in reverse
            d_rot1 = min(np.abs(Utils.angle_diff(d_rot1, 0.0)),
                         np.abs(Utils.angle_diff(d_rot1, np.pi)))
            d_rot2 = min(np.abs(Utils.angle_diff(d_rot2, 0.0)),
                         np.abs(Utils.angle_diff(d_rot2, np.pi)))

            # Debug hooks
            # print(f"dx={dx:.3f} | dy={dy:.3f} | dtheta={np.degrees(dtheta):.3f}")
            # print(f"d_rot1={np.degrees(d_rot1):.3f} | d_rot2={np.degrees(d_rot2):.3f} | d_trans={d_trans:.3f}")

            # TUM model's improvement
            if self.MOTION_MODEL == 'amcl':
                scale_rot1 = a1*d_rot1+a2*d_trans
                scale_rot2 = a1*d_rot2+a2*d_trans
            else:
                # print("d_trans:", d_trans, "v:", self.odom_msgs[-1].twist.twist.linear.x)
                scale_rot1 = a1*d_rot1+a2/(max(d_trans, self.LAM_THRESH))
                scale_rot2 = a1*d_rot2+a2/(max(d_trans, self.LAM_THRESH))

            scale_trans = a3*d_trans + a4*(d_rot1+d_rot2)

            # If we are reversing, add movement noise
            scale_rot1 *= reverse_spread
            scale_rot2 *= reverse_spread
            scale_trans *= reverse_spread

            d_rot1 += np.random.normal(scale=scale_rot1,
                                       size=self.MAX_PARTICLES)
            # It is more likely that we move forward, so shift the mean of the translation vector.
            # A choice of half a std-deviation is made here.
            d_trans += np.random.normal(loc=scale_trans/2, scale=scale_trans,
                                        size=self.MAX_PARTICLES)
            d_rot2 += np.random.normal(scale=scale_rot2,
                                       size=self.MAX_PARTICLES)

            # ? Future Extension: To add speed-dependent lateral offset to pose
            eff_hdg = proposal_dist[:, 2]+d_rot1+reverse_offset
            proposal_dist[:, 0] += d_trans*np.cos(eff_hdg)
            proposal_dist[:, 1] += d_trans*np.sin(eff_hdg)
            proposal_dist[:, 2] += d_rot1 + d_rot2
        elif self.MOTION_MODEL=='arc':
            dx_, dy_, dtheta_ = action

            scale_x = max(self.MOTION_DISPERSION_ARC_X_MIN, np.abs(dx_) * self.MOTION_DISPERSION_ARC_X)
            scale_y = np.abs(dy_) * self.MOTION_DISPERSION_ARC_Y
            # If above threshold add noise from x too
            if np.abs(dx_) > self.MOTION_DISPERSION_ARC_XY_MIN_X:
                scale_y += (np.abs(dx_)-self.MOTION_DISPERSION_ARC_XY_MIN_X)*self.MOTION_DISPERSION_ARC_XY
            scale_y = min(self.MOTION_DISPERSION_ARC_Y_MAX, max(self.MOTION_DISPERSION_ARC_Y_MIN, scale_y))
            scale_th = max(self.MOTION_DISPERSION_ARC_THETA_MIN, np.abs(dtheta_) * self.MOTION_DISPERSION_ARC_THETA)

            # debugging scales
            # rospy.loginfo(f"{np.abs(dx_)=:.4f} | {scale_x:.4f} | {self.last_odom_msg.twist.twist.linear.x:.4f}")
            # rospy.loginfo(f"{np.abs(dy_)=:.4f} | {scale_y:.4f}")
            # rospy.loginfo(f"{np.abs(dtheta_)=:.4f} | {scale_th:.4f}\n")

            dx = np.random.normal(loc=dx_, scale=scale_x, size=self.MAX_PARTICLES)
            dy = np.random.normal(loc=dy_, scale=scale_y, size=self.MAX_PARTICLES)
            dtheta = np.random.normal(loc=dtheta_, scale=scale_th, size=self.MAX_PARTICLES)

            r = dx/dtheta
            angle = (np.pi/2) * ( np.sign(dtheta)+(dtheta==0) )     # +90 if dtheta is positive
            cx = proposal_dist[:, 0] + r*np.cos(proposal_dist[:, 2] + angle)
            cy = proposal_dist[:, 1] + r*np.sin(proposal_dist[:, 2] + angle)

            psi = np.arctan2(proposal_dist[:, 1]-cy, proposal_dist[:, 0]-cx)

            x_ = cx + r*np.cos(psi + dtheta)
            y_ = cy + r*np.sin(psi + dtheta)
            theta_ = proposal_dist[:, 2] + dtheta

            x_ += dy*np.cos(theta_ + np.pi/2)
            y_ += dy*np.sin(theta_ + np.pi/2)

            proposal_dist[:, 0] = x_
            proposal_dist[:, 1] = y_
            proposal_dist[:, 2] = theta_

        else:
            dx, dy, dtheta = action

            # rotate the action into the coordinate space of each particle
            cosines = np.cos(proposal_dist[:, 2])
            sines = np.sin(proposal_dist[:, 2])

            self.local_deltas[:, 0] = cosines*dx - sines*dy
            self.local_deltas[:, 1] = sines*dx + cosines*dy
            self.local_deltas[:, 2] = dtheta

            proposal_dist[:, :] += self.local_deltas
            proposal_dist[:, 0] += np.random.normal(
                loc=0.0, scale=self.MOTION_DISPERSION_X, size=self.MAX_PARTICLES)
            proposal_dist[:, 1] += np.random.normal(
                loc=0.0, scale=self.MOTION_DISPERSION_Y, size=self.MAX_PARTICLES)
            proposal_dist[:, 2] += np.random.normal(
                loc=0.0, scale=self.MOTION_DISPERSION_THETA, size=self.MAX_PARTICLES)

        # clamp angles in proposal distribution around (-pi, pi)
        proposal_s = np.sin(proposal_dist[:,2])
        proposal_c = np.cos(proposal_dist[:,2])
        proposal_dist[:,2] = np.arctan2(proposal_s, proposal_c)

        if self.DEBUG:
            self.motion_model_time_ms.append((time()-tic)*1000)

        # ? Future Extension: resample if the particle goes out of track?

    def sensor_model(self, proposal_dist, obs, weights):
        '''
        This function computes a probablistic weight for each particle in the proposal distribution.
        These weights represent how probable each proposed (x,y,theta) pose is given the measured
        ranges from the lidar scanner.

        There are 4 different variants using various features of RangeLibc for demonstration purposes.
        - VAR_REPEAT_ANGLES_EVAL_SENSOR is the most stable, and is very fast.
        - VAR_NO_EVAL_SENSOR_MODEL directly indexes the precomputed sensor model. This is slow
                                   but it demonstrates what self.range_method.eval_sensor_model does
        - VAR_RADIAL_CDDT_OPTIMIZATIONS is only compatible with CDDT or PCDDT, it implments the radial
                                        optimizations to CDDT which simultaneously performs ray casting
                                        in two directions, reducing the amount of work by roughly a third
        '''
        if self.DEBUG:
            tic = time()
        num_rays = self.DES_LIDAR_BEAMS
        # only allocate buffers once to avoid slowness
        if self.first_sensor_update:
            if self.RANGELIB_VAR == RangeLibVariant.VAR_NO_EVAL_SENSOR_MODEL or \
               self.RANGELIB_VAR == RangeLibVariant.VAR_CALC_RANGE_MANY_EVAL_SENSOR:

                rospy.logwarn("""In these modes, the proposal_dist is not transformed from the base_links to the laser frame.
                              Performance of PF will be worse. Try using another variant instead, as they offer better speed anyway!""")

                self.queries = np.zeros(
                    (num_rays*self.MAX_PARTICLES, 3), dtype=np.float32)
            else:
                self.queries = np.zeros(
                    (self.MAX_PARTICLES, 3), dtype=np.float32)

            self.ranges = np.zeros(
                num_rays*self.MAX_PARTICLES, dtype=np.float32)
            self.tiled_angles = np.tile(
                self.LIDAR_THETA_LUT, self.MAX_PARTICLES)
            self.first_sensor_update = False

        # transform particles into the laser frame
        proposal_s = np.sin(proposal_dist[:,2])
        proposal_c = np.cos(proposal_dist[:,2])
        rot = np.array([[proposal_c, -proposal_s],
                        [proposal_s, proposal_c]]).transpose(2,0,1)   # (N,2,2)
        laser_offset_2d = self.laser_base_link_offset[:2]
        res = ( rot @ laser_offset_2d[np.newaxis, :, np.newaxis] ).reshape(self.MAX_PARTICLES, 2)

        self.queries[:, :] = proposal_dist
        self.queries[:, :2] += res

        # ! THIS assumes constant angle between scans but we don't do this.
        if self.RANGELIB_VAR == RangeLibVariant.VAR_RADIAL_CDDT_OPTIMIZATIONS:
            if "cddt" in self.WHICH_RM:
                # self.queries[:, :] = proposal_dist[:, :]
                self.range_method.calc_range_many_radial_optimized(
                    num_rays, self.downsampled_angles[0], self.downsampled_angles[-1], self.queries, self.ranges)

                # evaluate the sensor model
                self.range_method.eval_sensor_model(
                    obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
                # apply the squash factor
                self.weights = np.power(self.weights, self.INV_SQUASH_FACTOR)
            else:
                raise ValueError(
                    "Cannot use radial optimizations with non-CDDT based methods, use rangelib_variant 2")
        elif self.RANGELIB_VAR == RangeLibVariant.VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT:
            # self.queries[:, :] = proposal_dist[:, :]
            self.range_method.calc_range_repeat_angles_eval_sensor_model(
                self.queries, self.LIDAR_THETA_LUT, obs, self.weights)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == RangeLibVariant.VAR_REPEAT_ANGLES_EVAL_SENSOR:
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            # self.queries[:, :] = proposal_dist[:, :]
            self.range_method.calc_range_repeat_angles(
                self.queries, self.LIDAR_THETA_LUT, self.ranges)

            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(
                obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == RangeLibVariant.VAR_CALC_RANGE_MANY_EVAL_SENSOR:
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            # this part is inefficient since it requires a lot of effort to construct this redundant array
            self.queries[:, 0] = np.repeat(proposal_dist[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(proposal_dist[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(proposal_dist[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles

            self.range_method.calc_range_many(self.queries, self.ranges)

            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(
                obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == RangeLibVariant.VAR_NO_EVAL_SENSOR_MODEL:
            # this version directly uses the sensor model in Python, at a significant computational cost
            self.queries[:, 0] = np.repeat(proposal_dist[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(proposal_dist[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(proposal_dist[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles

            # compute the ranges for all the particles in a single functon call
            self.range_method.calc_range_many(self.queries, self.ranges)

            # resolve the sensor model by discretizing and indexing into the precomputed table
            obs /= float(self.map_info.resolution)
            ranges = self.ranges / float(self.map_info.resolution)
            obs[obs > self.MAX_RANGE_PX] = self.MAX_RANGE_PX
            ranges[ranges > self.MAX_RANGE_PX] = self.MAX_RANGE_PX

            intobs = np.rint(obs).astype(np.uint16)
            intrng = np.rint(ranges).astype(np.uint16)

            # compute the weight for each particle
            for i in range(self.MAX_PARTICLES):
                weight = np.product(
                    self.sensor_model_table[intobs, intrng[i*num_rays:(i+1)*num_rays]])
                weight = np.power(weight, self.INV_SQUASH_FACTOR)
                weights[i] = weight
        else:
            raise ValueError(
                f"Please set rangelib_variant param to 0-4. Current value: {self.RANGELIB_VAR}")

        if self.DEBUG:
            self.sensor_model_time_ms.append((time()-tic)*1000)

    def MCL(self, odom_data, observations):
        '''
        Performs one step of Monte Carlo Localization.
            1. resample particle distribution to form the proposal distribution
            2. apply the motion model
            3. apply the sensor model
            4. normalize particle weights
        '''
        # draw the proposal distribution from the old particles
        proposal_indices = np.random.choice(
            self.particle_indices, self.MAX_PARTICLES, p=self.weights)
        proposal_distribution = self.particles[proposal_indices, :]

        # compute the motion model to update the proposal distribution
        self.motion_model(proposal_distribution, odom_data)

        # compute the sensor model
        self.sensor_model(proposal_distribution, observations, self.weights)

        # check for permissible region and downscale weight if a particle goes out of track
        # ? Future Extension: express map in 'world' coordinates to save on this optimization?
        particles_in_map = np.copy(proposal_distribution)
        Utils.world_to_map(particles_in_map, self.map_info)
        limit = self.permissible_region.shape
        particles_in_map = np.clip(particles_in_map[:, 0:2].astype('int'),
                                   [0, 0], [limit[1]-1, limit[0]-1])
        valid_particles = self.permissible_region[particles_in_map[:,
                                                                   1], particles_in_map[:, 0]]
        self.weights = np.where(
            valid_particles, self.weights, 0.01*self.weights)

        # normalize importance weights
        weight_sum = np.sum(self.weights)

        # Empirically tuned term
        if False:
        # if weight_sum < 1e-16:
            # ? Future Extension: Send messages somewhere to alert a safety controller
            rospy.logerr("Particle depletion occured!")
            # First release the state lock to effect change in global init
            self.state_lock.release()
            self.initialize_global()
            self.state_lock.acquire()
        else:
            self.weights /= weight_sum

        # save the particles
        self.particles = proposal_distribution

        # Compute particle covariance about the inferred pose
        if self.inferred_pose is not None and self.PUB_COVARIANCE:
            spread = self.particles - self.inferred_pose    #(N,3)
            # Distance in theta calculation to be (-pi,pi)
            spread[spread[:,2]> np.pi, 2] -= 2*np.pi
            spread[spread[:,2]<-np.pi, 2] += 2*np.pi

            spread = spread[:, :, np.newaxis]                               # (N,3,1)
            inner_prod = spread @ spread.transpose(0, 2, 1)                 # (N,3,1) @ (N,1,3) = (N,3,3)
            res = self.weights[:, np.newaxis, np.newaxis] * inner_prod      # (N,1,1)*(N,3,3) = (N,3,3)
            self.cov = np.sum(res, axis=0)                                  # (3,3)

    def update(self):
        '''
        Apply the MCL function to update particle filter state.

        Ensures the state is correctly initialized, and acquires the state lock before proceeding.
        '''
        if not (self.lidar_initialized and self.odom_initialized and self.map_initialized):
            return

        if self.state_lock.locked():
            rospy.loginfo("Concurrency error avoided")
            return

        self.state_lock.acquire()

        observation = np.copy(self.downsampled_ranges).astype(np.float32)

        # run the MCL update algorithm
        self.MCL(self.odometry_data, observation)
        self.odometry_data = np.zeros(3)

        # compute the expected value of the robot pose
        inferred_x = np.sum(self.particles[:,0] * self.weights)
        inferred_y = np.sum(self.particles[:,1] * self.weights)
        inferred_s = np.sum(np.sin(self.particles[:,2]) * self.weights)
        inferred_c = np.sum(np.cos(self.particles[:,2]) * self.weights)

        self.inferred_pose = np.array(
            ( inferred_x, inferred_y , np.arctan2(inferred_s, inferred_c) )
        ) # (3,)

        self.state_lock.release()

        # publish transformation frame based on inferred pose
        self.publish_tf(self.inferred_pose, self.last_stamp)

        self.visualise()

    # Main loop
    def loop(self):
        while not rospy.is_shutdown():
            if self.DEBUG:
                self.profiler.enable()
                tic = time()

            # Assert odom has been initialised / there are things to process
            if len(self.odom_msgs) == 0:
                continue

            # Get the furthest-back message in the buffer
            msg: Odometry = self.odom_msgs[0]

            # Quantify gap in message pubs (this was solved with the tcp_nodelay setting)
            # msg_time = msg.header.stamp.to_sec()
            # now_time = rospy.Time.now().to_sec()
            # rospy.loginfo(f"Time gap to present: {(now_time-msg_time)*1000:.2f}ms | Buffer Size: {len(self.odom_msgs)}")

            position = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y])

            orientation = Utils.quaternion_to_angle(msg.pose.pose.orientation)
            self.curr_pose = np.array([position[0], position[1], orientation])

            self.update()   # Update based on curr_pose and last_pose

            if isinstance(self.last_pose, np.ndarray):
                # changes in x,y,theta in local coordinate system of the car
                rot = Utils.rotation_matrix(-self.last_pose[2])
                delta = np.array(
                    [position - self.last_pose[0:2]]).transpose()
                local_delta = (rot*delta).transpose()
                self.odometry_data = np.array(
                    [local_delta[0, 0], local_delta[0, 1], orientation - self.last_pose[2]])

                self.odom_initialized = True
            else:
                rospy.loginfo("PF2...Received first Odometry message")

            # self.last_stamp = msg.header.stamp
            self.last_stamp = rospy.Time.now()
            self.last_pose = self.curr_pose
            self.odom_msgs.popleft()

            self.rate.sleep()

            if self.DEBUG:
                self.profiler.disable()
                self.overall_time_ms.append((time()-tic)*1000)

                if (self.itr % 30) == 0:
                    s_mean = np.mean(self.sensor_model_time_ms)
                    s_std = np.std(self.sensor_model_time_ms)
                    m_mean = np.mean(self.motion_model_time_ms)
                    m_std = np.std(self.motion_model_time_ms)
                    o_mean = np.mean(self.overall_time_ms)
                    o_std = np.std(self.overall_time_ms)

                    rospy.loginfo(
                        f"Sensor Model: {s_mean:4.2f}ms std:{s_std:4.2f}ms | Motion Model: {m_mean:4.2f}ms std:{m_std:4.2f}ms | Overall: {o_mean:4.2f}ms std:{o_std:4.2f}ms")

                if (self.itr % 500) == 0:
                    stats = pstats.Stats(self.profiler)
                    stats.sort_stats(pstats.SortKey.TIME)
                    # look for this in ~/.ros
                    stats.dump_stats(filename="pf2_stats.prof")
                    rospy.logwarn("PF2 Dumping profiling stats to file.")

                self.itr += 1


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    pf.loop()
    rospy.spin()

# // Put this into __init__()
# // To decouple publishing a pose from computation of inferred pose...
#         self.PUBLISH_RATE = 100
#         rospy.Timer(rospy.Duration(1/self.PUBLISH_RATE), self.publish_inferred_pose)

# // Set up this callback for various types of upsampling + interpolation
# // Option 1: `deadreckon` does not do very well, increases the amount of jitter
# // Option 2: `spline-smoothing` also does not seem super promising.
#     # This is experimental.
#     def publish_inferred_pose(self, timer: rospy.timer.TimerEvent):

#         # This is to alleviate "repeat transform" warnings especially when run with rosbags
#         now_time : float = rospy.get_time()
#         now_time_ros = rospy.Time.now()
#         if timer.last_real is not None:
#             time_slip = timer.current_real.to_sec() - timer.last_real.to_sec()
#             if time_slip > 1.2/self.PUBLISH_RATE or time_slip < 0.8/self.PUBLISH_RATE:
#                 return

#         # Several options:
#         # 0. (default: Do not attempt any interpolation.)
#         # '''
#         if self.inferred_pose is not None and \
#             not np.allclose(0,
#                             np.abs(self.last_odom_msg.header.stamp.to_sec()-self.last_pub_stamp.to_sec()) ):
#             self.publish_tf(self.inferred_pose, now_time_ros)
#         # '''

#         # 1. From last odom message, perform dead-reckoning of current pose.
#         # works quite well, <smooth> and ~70hz
#         '''
#         if self.inferred_pose is not None:
#             msg = self.last_odom_msg

#             delta_t = now_time - msg.header.stamp.to_sec()
#             # delta_t = now_time - self.last_stamp.to_sec()
#             # rospy.loginfo(f"{( msg.header.stamp.to_sec()-self.last_stamp.to_sec() )*1000}ms")
#             # rospy.loginfo(f"Now: {str(now_time)[4:]} Diff in time from last odom to now: {delta_t*1000:.3f}ms. Curr Exp: {timer.current_expected.to_sec()} Curr Real: {timer.current_real.to_sec()}")
#             delta_x_B = msg.twist.twist.linear.x * delta_t
#             delta_y_B = msg.twist.twist.linear.y * delta_t
#             delta_th_B = msg.twist.twist.angular.z * delta_t

#             deadreckon_pose = [
#                 self.inferred_pose[0] + delta_x_B*np.cos(self.inferred_pose[2]) - delta_y_B*np.sin(self.inferred_pose[2]),
#                 self.inferred_pose[1] + delta_x_B*np.sin(self.inferred_pose[2]) + delta_y_B*np.cos(self.inferred_pose[2]),
#                 self.inferred_pose[2] + delta_th_B,
#             ]

#             self.publish_tf(deadreckon_pose, now_time_ros)
#         '''

#         # 2. From past odom messages compute spline for smoothing. Extrapolation is unstable, do NOT attempt it!
#         # With spline buffer of length 5, (smoothing=4) it takes about 10-50ms to execute -> 20-25Hz output.
#         # == high compute and smoothing not exactly visible
#         '''
#         tic = time()
#         if len(self.t_history_inferred) > 3 and not self.state_lock.locked():
#             delta_t = now_time - self.t_history_inferred[-1]
#             # rospy.loginfo(f"Now: {str(now_time)[4:]} Diff in time from last odom to now: {delta_t*1000:.3f}ms.")

#             assert len(self.t_history_inferred)==len(self.x_history_inferred)
#             tck_x = splrep(self.t_history_inferred, self.x_history_inferred, s=len(self.t_history_inferred)-1)
#             tck_y = splrep(self.t_history_inferred, self.y_history_inferred, s=len(self.t_history_inferred)-1)
#             tck_s = splrep(self.t_history_inferred, self.s_history_inferred, s=len(self.t_history_inferred)-1)
#             tck_c = splrep(self.t_history_inferred, self.c_history_inferred, s=len(self.t_history_inferred)-1)

#             # smoothing
#             x_int = BSpline(*tck_x)(self.t_history_inferred[-1])
#             y_int = BSpline(*tck_y)(self.t_history_inferred[-1])
#             s_int = BSpline(*tck_s)(self.t_history_inferred[-1])
#             c_int = BSpline(*tck_c)(self.t_history_inferred[-1])

#             rospy.loginfo(f"{x_int=:.2f}, {self.x_history_inferred[-1]=:.2f}, diff:{x_int-self.x_history_inferred[-1]:.2f}")
#             rospy.loginfo(f"{y_int=:.2f}, {self.y_history_inferred[-1]=:.2f}, diff:{y_int-self.y_history_inferred[-1]:.2f}")
#             rospy.loginfo(f"Comp_time = {(time()-tic)*1000}ms \n")

#             deadreckon_pose = [ x_int, y_int, np.arctan2(s_int, c_int) ]
#             self.publish_tf(deadreckon_pose, rospy.Time.now())
#         '''