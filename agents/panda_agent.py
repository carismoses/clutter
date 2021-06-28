import sys
import time
import numpy
import matplotlib.pyplot as plt
import pb_robot
import pyquaternion
import pybullet as p
import random
from copy import deepcopy

from block_utils import get_adversarial_blocks, rotation_group, ZERO_POS, \
                        Quaternion, get_rotated_block, Pose, add_noise, \
                        Environment, Position, World
from pddlstream.utils import INF
from pybullet_utils import transformation
import tamp.primitives
from tamp.misc import setup_panda_world, get_pddl_block_lookup, \
                      print_planning_problem, ExecuteActions, ExecutionFailure
from tamp.pddlstream_utils import get_pddlstream_info, pddlstream_plan

class PandaAgent:
    def __init__(self, blocks, noise=0.00005, block_init_xy_poses=None,
                 use_platform=False, use_vision=False, real=False,
                 use_planning_server=False, use_learning_server=False, 
                 alternate_orientations=False, task='stacking'):
        """
        Build the Panda world in PyBullet and set up the PDDLStream solver.
        The Panda world should in include the given blocks as well as a
        platform which can be used in experimentation.
        :param use_platform: Boolean stating whether to include the platform to
                             push blocks off of or not.
        :param use_vision: Boolean stating whether to use vision to detect blocks.
        :param use_planning_server: Boolean stating whether to use the separate
                                    ROS planning service server.
        :param use_learning_server: Boolean stating whether to host a ROS service
                                    server to drive planning from active learning script.
        :param alternate_orientations: Boolean stating whether blocks can be replaced in 
                                       their home positions at alternate orientations.

        If you are using the ROS action server, you must start it in a separate terminal:
            rosrun stacking_ros planning_server.py
        """
        self.real = real
        self.use_vision = use_vision
        self.use_platform = use_platform
        self.use_planning_server = use_planning_server
        self.use_learning_server = use_learning_server
        self.alternate_orientations = alternate_orientations

        # Setup PyBullet instance to run in the background and handle planning/collision checking.
        self._planning_client_id = pb_robot.utils.connect(use_gui=False)
        self.plan()
        pb_robot.utils.set_default_camera()
        self.robot = pb_robot.panda.Panda()
        self.robot.arm.hand.Open()
        self.belief_blocks = blocks
        self.pddl_blocks, self.platform_table, self.platform_leg, self.table, self.frame, self.wall = setup_panda_world(self, self.robot,
                                                                                                        blocks,
                                                                                                        block_init_xy_poses,
                                                                                                        use_platform=use_platform,
                                                                                                        task=task,
                                                                                                        client='planning')
        self.fixed = [self.platform_table, self.platform_leg, self.table, self.frame, self.wall]
        
        self.pddl_block_lookup = get_pddl_block_lookup(blocks, self.pddl_blocks)

        self.orig_joint_angles = self.robot.arm.GetJointValues()
        self.orig_block_poses = [b.get_base_link_pose() for b in self.pddl_blocks]

        # Setup PyBullet instance that only visualizes plan execution. State needs to match the planning instance.
        poses = [b.get_base_link_pose() for b in self.pddl_blocks]
        poses = [Pose(Position(*p[0]), Quaternion(*p[1])) for p in poses]
        self._execution_client_id = pb_robot.utils.connect(use_gui=False)
        self.execute()
        pb_robot.utils.set_default_camera()
        self.execution_robot = pb_robot.panda.Panda()
        self.execution_robot.arm.hand.Open()
        setup_panda_world(self, 
                            self.execution_robot, 
                            blocks, 
                            poses, 
                            use_platform=use_platform, 
                            task=task,
                            client='execution')
        self.successful_contact_data = []
        self.failed_contact_data = []
        self.successful_closest_data = []
        self.failed_closest_data =[]
        
        # Set up ROS plumbing if using features that require it
        if self.use_vision or self.use_planning_server or self.use_learning_server or real:
            import rospy
            try:
                rospy.init_node("panda_agent")
            except:
                print("ROS Node already created")

        # Create an arm interface
        if real:
            from franka_interface import ArmInterface
            self.real_arm = ArmInterface()

            from franka_core_msgs.msg import RobotState
            state_topic = "/franka_ros_interface/custom_franka_state_controller/robot_state"
            self.arm_last_error_time = time.time()
            self.arm_error_check_time = 3.0
            self.arm_state_subscriber = rospy.Subscriber(
                state_topic, RobotState, self.robot_state_callback)

                
        # step sim to let blocks settle when simulating cluttered world
        self.step_simulation(1000, vis_frames=False)

        # Set initial poses of all blocks and setup vision ROS services.
        if self.use_vision:
            from panda_vision.srv import GetBlockPosesWorld, GetBlockPosesWrist
            rospy.wait_for_service('get_block_poses_world')
            rospy.wait_for_service('get_block_poses_wrist')
            self._get_block_poses_world = rospy.ServiceProxy('get_block_poses_world', GetBlockPosesWorld)
            self._get_block_poses_wrist = rospy.ServiceProxy('get_block_poses_wrist', GetBlockPosesWrist)
        # Start ROS clients and servers as needed
        self.last_obj_held = None
        if self.use_planning_server:
            from stacking_ros.srv import GetPlan, SetPlanningState
            from tamp.ros_utils import goal_to_ros, ros_to_task_plan

            print("Waiting for planning server...")
            rospy.wait_for_service("get_latest_plan")
            self.goal_to_ros = goal_to_ros
            self.ros_to_task_plan = ros_to_task_plan
            self.init_state_client = rospy.ServiceProxy(
                "/reset_planning", SetPlanningState)
            self.get_plan_client = rospy.ServiceProxy(
                "/get_latest_plan", GetPlan)
            print("Done!")
        if self.use_learning_server:
            from stacking_ros.srv import PlanTower
            self.learning_server = rospy.Service(
                "/plan_tower", PlanTower, self.learning_server_callback)
            print("Learning server started!")

        self.pddl_info = get_pddlstream_info(self.robot,
                                             self.fixed,
                                             self.pddl_blocks,
                                             add_slanted_grasps=False,
                                             approach_frame='global',
                                             use_vision=self.use_vision)

        self.noise = noise
        self.txt_id = None
        self.plan()
        self.task = task

    def _add_text(self, txt):
        self.execute()
        pb_robot.viz.remove_all_debug()
        self.txt_id = pb_robot.viz.add_text(txt, position=(0, 0.25, 0.75), size=2)
        self.plan()


    def execute(self):
        self.state = 'execute'
        pb_robot.aabb.set_client(self._execution_client_id)
        pb_robot.body.set_client(self._execution_client_id)
        pb_robot.collisions.set_client(self._execution_client_id)
        pb_robot.geometry.set_client(self._execution_client_id)
        pb_robot.grasp.set_client(self._execution_client_id)
        pb_robot.joint.set_client(self._execution_client_id)
        pb_robot.link.set_client(self._execution_client_id)
        pb_robot.panda.set_client(self._execution_client_id)
        pb_robot.planning.set_client(self._execution_client_id)
        pb_robot.utils.set_client(self._execution_client_id)
        pb_robot.viz.set_client(self._execution_client_id)


    def plan(self):
        if self.use_planning_server:
            return
        self.state = 'plan'
        pb_robot.aabb.set_client(self._planning_client_id)
        pb_robot.body.set_client(self._planning_client_id)
        pb_robot.collisions.set_client(self._planning_client_id)
        pb_robot.geometry.set_client(self._planning_client_id)
        pb_robot.grasp.set_client(self._planning_client_id)
        pb_robot.joint.set_client(self._planning_client_id)
        pb_robot.link.set_client(self._planning_client_id)
        pb_robot.panda.set_client(self._planning_client_id)
        pb_robot.planning.set_client(self._planning_client_id)
        pb_robot.utils.set_client(self._planning_client_id)
        pb_robot.viz.set_client(self._planning_client_id)


    def reset_world(self):
        """ Resets the planning world to its original configuration """
        print("Resetting world")

        if self.real:
            angles = self.real_arm.convertToList(self.real_arm.joint_angles())
        else:
            angles = self.orig_joint_angles
        self.plan()
        self.robot.arm.SetJointValues(angles)
        self.execute()
        self.execution_robot.arm.SetJointValues(angles)
        for bx, b in enumerate(self.pddl_blocks):
            b.set_base_link_pose(self.orig_block_poses[bx])
        print("Done")


    def _get_initial_pddl_state(self):
        """
        Get the PDDL representation of the world between experiments. This
        method assumes that all blocks are on the table. We will always "clean
        up" an experiment by moving blocks away from the platform after an
        experiment.
        """
        fixed = [self.table, self.platform_table, self.platform_leg, self.frame]
        conf = pb_robot.vobj.BodyConf(self.robot, self.robot.arm.GetJointValues())
        print('Initial configuration:', conf.configuration)
        init = [('CanMove',),
                ('Conf', conf),
                ('StartConf', conf),
                ('AtConf', conf),
                ('HandEmpty',)]

        self.table_pose = pb_robot.vobj.BodyPose(self.table, self.table.get_base_link_pose())
        init += [('Pose', self.table, self.table_pose), 
                 ('AtPose', self.table, self.table_pose)]

        for body in self.pddl_blocks:
            print(type(body), body)
            pose = pb_robot.vobj.BodyPose(body, body.get_base_link_pose())
            init += [('Graspable', body),
                    ('Pose', body, pose),
                    ('AtPose', body, pose),
                    ('Block', body),
                    ('On', body, self.table),
                    ('Supported', body, pose, self.table, self.table_pose)]

        if not self.platform_table is None:
            platform_pose = pb_robot.vobj.BodyPose(self.platform_table, self.platform_table.get_base_link_pose())
            init += [('Pose', self.platform_table, platform_pose), 
                    ('AtPose', self.platform_table, platform_pose)]
            init += [('Block', self.platform_table)]
        init += [('Table', self.table)]
        return init



    def build_reset_problem(self):
        """ Builds the initial conditions for a tower reset given a set of moved blocks """

        print("Resetting blocks...")
        print("Moved Blocks:", self.moved_blocks)
        
        # Define block order by random shuffle
        block_ixs = numpy.arange(len(self.pddl_blocks))
        numpy.random.shuffle(block_ixs)
        # Build the initial data structures
        if self.use_planning_server:
            from stacking_ros.msg import BodyInfo
            from stacking_ros.srv import SetPlanningStateRequest
            from tamp.ros_utils import block_init_to_ros, pose_to_ros, pose_tuple_to_ros, transform_to_ros
            ros_req = SetPlanningStateRequest()
            ros_req.init_state = block_init_to_ros(self.pddl_blocks)
            if self.real:
                ros_req.robot_config.angles = self.real_arm.convertToList(self.real_arm.joint_angles())
            else:
                ros_req.robot_config.angles = self.robot.arm.GetJointValues()
        else:
            pddl_problems = []

        # TODO: get home positions from somewhere else and reset this so compatible with stacking
        storage_poses = [(-0.4, -0.45), (-0.4, -0.25), # Left Corner
                         (-0.25, -0.5), (-0.4, 0.25),   # Back Center
                         (-0.4, 0.45), (-0.25, 0.5),   # Right Corner
                         (-0., -0.5), (0., -0.35),   # Left Side
                         (-0., 0.5), (0., 0.35)]     # Right Side
        if self.task == 'clutter':
            self.original_poses = []
            for ix, block in enumerate(self.pddl_blocks):
                self.original_poses.append(((storage_poses[ix][0], storage_poses[ix][1], 0.0),
                                        (0., 0., 0., 1.)))
            
        # Add all blocks to be moved to the data structure
        for ix in block_ixs:
            blk, pose = self.pddl_blocks[ix], self.original_poses[ix]
            if blk in self.moved_blocks:
                if self.use_planning_server:
                    goal_pose = pb_robot.vobj.BodyPose(blk, pose)
                    block_ros = BodyInfo()
                    block_ros.name = blk.readableName
                    block_ros.stack = False
                    pose_to_ros(goal_pose, block_ros.pose)
                    ros_req.goal_state.append(block_ros)
                else:
                    pddl_problems.append((self.table, blk, pose))

        # Return the planning data structure
        if self.use_planning_server:
            return ros_req
        else:
            return pddl_problems


    def simulate_clutter(self, vis, T=2500, real=False, base_xy=(0., 0.5), ignore_resets=False):
        """
        Simulates a scene of randomly generated clutter

        Returns:
          success : Flag indicating success of execution (True/False)
        """
        self.moved_blocks = self.pddl_blocks

        # for block in self.pddl_blocks:
        #     print('pose is ...', self.orig_block_poses)

        planning_probs = self.build_reset_problem()
        

        for planning_prob in planning_probs:
            print('planning problem', planning_prob)
            self.plan_and_execute([planning_prob], real, T)


    def plan_and_execute(self, planning_prob, real=False, T=2500):
        """
        Requests a PDDLStream plan from a planning server and executes the resulting plan
        Parameters:
            planning_prob : Planning problem 
            real : Flag for whether we are executing on the real panda robot
            T : Number of time steps to step simulator to see if tower has fallen
        """

        problem_ix = 0

        # PLANNING
        
        base, blk, pose = planning_prob[problem_ix]
        query_block = blk

        self.step_simulation(T, vis_frames=False)

        contact_points = []
        for blkB in self.pddl_blocks:
            contact_points += p.getContactPoints(blk.id, blkB.id)
        closest_points = []
        for blkB in self.pddl_blocks:
            closest_points += p.getClosestPoints(blk.id, blkB.id, 0.1)

        print('contact points are:', contact_points)
        print(len(contact_points))
        # print('closest points are:', closest_points)

        p.changeVisualShape(blk.id, blk.base_link, rgbaColor = [0, 0, 0, 1])
        view_matrix = p.computeViewMatrixFromYawPitchRoll(distance=1.0,
                                                          yaw=90,
                                                          pitch=-45,
                                                          roll=0,
                                                          upAxisIndex=2,
                                                          cameraTargetPosition=(0., 0., 0.5))
        aspect = 100. / 190.
        nearPlane = 0.01
        farPlane = 10
        fov = 90
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        image_data = p.getCameraImage(400, 400, shadow=1, viewMatrix=view_matrix,
                                      projectionMatrix=projection_matrix)
        w, h, im = image_data[:3]
        np_im = numpy.array(im, dtype=numpy.uint8).reshape(h, w, 4)[:, :, 0:3]
        # plt.imshow(numpy.array(np_im))
        # plt.ion()
        # plt.show()

        self._add_text('Planning block placement')
        self.plan()
        saved_world = pb_robot.utils.WorldSaver()
        self.robot.arm.hand.Open()
        
        # Unpack initial conditions
        fixed_objs = self.fixed + [b for b in self.pddl_blocks if b != blk]
        init = self._get_initial_pddl_state()
        goal_terms = []
        # condition for returning block to home position
        init += [("Reset",)]
        goal_terms.append(("AtHome", blk))
        goal = tuple(['and'] + goal_terms)
        
        # Plan with PDDLStream
        pddl_info = get_pddlstream_info(self.robot,
                                        fixed_objs,
                                        self.pddl_blocks,
                                        add_slanted_grasps=True,
                                        approach_frame='global',
                                        use_vision=self.use_vision,
                                        home_pose=pose)
        plan, cost = pddlstream_plan(pddl_info, init, goal, 
                                     search_sample_ratio=1.0, 
                                     max_time=INF)
        if plan is None:
            print("\nFailed to plan\n")
            print("before length", contact_points)
            self.failed_contact_data.append(len(contact_points))
            self.failed_closest_data.append(len(closest_points))
            return
        saved_world.restore() # unsure about this line

        print("\nGot plan:")
        print(plan)
        print("before length", contact_points)
        self.successful_contact_data.append(len(contact_points))
        self.successful_closest_data.append(len(closest_points))

        # Once we have a plan, execute it
        obstacles = [f for f in self.fixed if f is not None]
        self.plan()
        ExecuteActions(plan, real=False, pause=False, wait=False, obstacles=obstacles)
        self.execute()
        ExecuteActions(plan, real=real, pause=True, wait=False, prompt=False, obstacles=obstacles, 
                       sim_fatal_failure_prob=0.0, sim_recoverable_failure_prob=0.0)

        self.step_simulation(T, vis_frames=False)

    def step_simulation(self, T, vis_frames=False, lifeTime=0.1, client='both'):
        if client == 'planning':
            p.setGravity(0, 0, -10, physicsClientId=self._planning_client_id)

            q = self.robot.get_joint_positions()

            for _ in range(T):
                p.stepSimulation(physicsClientId=self._planning_client_id)
                self.plan()
                self.robot.set_joint_positions(self.robot.joints, q)

                time.sleep(1 / 2400.)
        elif client == 'execution':
            p.setGravity(0, 0, -10, physicsClientId=self._execution_client_id)

            q = self.robot.get_joint_positions()

            for _ in range(T):
                p.stepSimulation(physicsClientId=self._execution_client_id)
                self.plan()
                self.robot.set_joint_positions(self.robot.joints, q)

                time.sleep(1 / 2400.)
        else:
            p.setGravity(0, 0, -10, physicsClientId=self._execution_client_id)
            p.setGravity(0, 0, -10, physicsClientId=self._planning_client_id)

            q = self.robot.get_joint_positions()
            for _ in range(T):
                p.stepSimulation(physicsClientId=self._execution_client_id)
                p.stepSimulation(physicsClientId=self._planning_client_id)

                self.execute()
                self.execution_robot.set_joint_positions(self.robot.joints, q)
                self.plan()
                self.robot.set_joint_positions(self.robot.joints, q)

                time.sleep(1/2400.)

                if vis_frames:
                    length = 0.1
                    for pddl_block in self.pddl_blocks:
                        pos, quat = pddl_block.get_pose()
                        new_x = transformation([length, 0.0, 0.0], pos, quat)
                        new_y = transformation([0.0, length, 0.0], pos, quat)
                        new_z = transformation([0.0, 0.0, length], pos, quat)

                        p.addUserDebugLine(pos, new_x, [1,0,0], lineWidth=3, lifeTime=lifeTime, physicsClientId=self._execution_client_id)
                        p.addUserDebugLine(pos, new_y, [0,1,0], lineWidth=3, lifeTime=lifeTime, physicsClientId=self._execution_client_id)
                        p.addUserDebugLine(pos, new_z, [0,0,1], lineWidth=3, lifeTime=lifeTime, physicsClientId=self._execution_client_id)