import pybullet as p
import time
import pybullet_data

import cv2
import numpy as np

JOINT_LIMITS = {
    "base_neck1": [-1.3, 1.3, 0],
    "neck1_neck2": [-0.65, 0.65, 0],
    "neck2_head": [-0.5, 0.5, 0],
    "R_eye_yaw": [-1, 1, 0],
    "R_eye_pitch": [-0.8, 0.8, 0],
    "L_eye_yaw": [-1, 1, 0],
    "L_eye_pitch": [-0.8, 0.8, 0],
    "Ruppereyelid": [-1, 0.5, -0.3],
    "Luppereyelid": [-1, 0.5, -0.3],
    "Rlowereyelid": [-0.5, 0.5, -0.1],
    "Llowereyelid": [-0.5, 0.5, -0.1],
}


class FaceRobotEnv:
    def __init__(self,
                 urdf_path="face_sim/urdf/face_sim.urdf",
                 use_gui=True,
                 debug_tool=True,
                 time_step=1. / 240.,object_location=[-0.25, -1, 0.5],usefixedbase=True, usecamera=True):
        """
        Initialize the Face Robot Environment.

        :param urdf_path: Path to the URDF file of the robot.
        :param use_gui: Whether to use the PyBullet GUI.
        :param debug_tool: Whether to enable debug sliders for continuous joints.
        :param time_step: The simulation time step.
        """
        self.use_gui = use_gui
        self.debug_tool = debug_tool
        self.time_step = time_step
        self.physics_client = p.connect(p.GUI if self.use_gui else p.DIRECT)
        self.usecamera=usecamera

        # Add default search path for plane.urdf, etc.
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the environment
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
        p.setGravity(0, 0, -9.81)

        # Load the robot
        start_pos = [0, 0, 0.3]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(urdf_path, start_pos, start_orn, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)


        # Set the GUI view window position and orientation
        if self.use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=2,  # Distance from the object
                cameraYaw=10,  # Rotate 10 degrees horizontally
                cameraPitch=-20,  # Look down at -20 degrees
                cameraTargetPosition=[0, 0, 0.5]  # Focus on the robot
            )

        # Initialize debug sliders for continuous joints
        self.joint_sliders = {}
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]

            if joint_type == p.JOINT_REVOLUTE and joint_name in JOINT_LIMITS:
                # Get the normalized initial position
                initial_real_value = JOINT_LIMITS[joint_name][2]
                initial_normalized = self.inverse_scale_action(joint_name, initial_real_value)

                # Create a debug slider with initial position set
                slider = p.addUserDebugParameter(joint_name, 0.0, 1.0, initial_normalized)
                self.joint_sliders[i] = slider

                # Set the initial joint position in the simulator
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=initial_real_value
                )

#camera and eye ball focus code
#############################################################################################################
        self.link_index=[]
    
        for i in range(self.num_joints):
            self.joint_info = p.getJointInfo(self.robot_id, i)

            if self.joint_info[12]==b'R_iris':
                self.link_index.append(i)

            elif self.joint_info[12]==b'L_iris':
                self.link_index.append(i)

        self.cube_start_position= object_location
        self.objectfixedbase = usefixedbase
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0]) 
        self.cube_id = p.loadURDF("cube_small.urdf", self.cube_start_position, cube_start_orientation, globalScaling=3, useFixedBase=self.objectfixedbase)

        self.cam_width = 400
        self.cam_height = 300
        self.cam_fov = 70
        self.cam_aspect = self.cam_width / self.cam_height
        self.cam_near = 0.02
        self.cam_far = 8

        self.camera_update_interval = 5  
        self.eyetracking_update_interval = 1
        self.step_count = 0


    def camera(self):

        for i in self.link_index:
            link_state=p.getLinkState(self.robot_id, i-2, computeForwardKinematics=True)
            position=np.array(link_state[0])


            up_vector =[0, 0, 1] 


            cube_state = p.getBasePositionAndOrientation(self.cube_id)
            cube_position = np.array(cube_state[0])

            target_position = cube_position
            
            view_matrix =p.computeViewMatrix(position, target_position, up_vector)
            projection_matrix =p.computeProjectionMatrixFOV(self.cam_fov, self.cam_aspect, self.cam_near, self.cam_far)
            img_data =p.getCameraImage(self.cam_width, self.cam_height, view_matrix, projection_matrix)


            rgb_img =np.reshape(img_data[2], (self.cam_height, self.cam_width, 4))[:, :, :3] 
            rgb_img =cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            cv2.imshow(f"Camera View {i-2}", rgb_img)

        cv2.waitKey(1)
    
    def eye_tracking(self): 

        neck_yaw_min, neck_yaw_max, _ = JOINT_LIMITS["base_neck1"]
        neck_pitch_min, neck_pitch_max, _ = JOINT_LIMITS["neck1_neck2"]

        eye_yaw_min, eye_yaw_max, _ = JOINT_LIMITS["L_eye_yaw"]
        eye_pitch_min, eye_pitch_max, _ = JOINT_LIMITS["L_eye_pitch"]

        neck_pitch=0
        neck_yaw=0
        
        cube_state = p.getBasePositionAndOrientation(self.cube_id)
        cube_position = np.array(cube_state[0])

        for i in self.link_index:
            link_state=p.getLinkState(self.robot_id, i-2, computeForwardKinematics=True)
            position=np.array(link_state[0])
            

            direction=cube_position-position
            direction_norm=direction/np.linalg.norm(direction)
            
            eye_yaw = np.arctan2(direction_norm[0], -direction_norm[1])
            eye_pitch = np.arcsin(direction_norm[2])

            if direction_norm[1] <=0 or direction_norm[1] > 0:

                if eye_yaw >= eye_yaw_max :
                    neck_yaw= eye_yaw - eye_yaw_max
                    eye_yaw = np.clip(eye_yaw, eye_yaw_min, eye_yaw_max)
                    neck_yaw= np.clip(neck_yaw, neck_yaw_min, neck_yaw_max)

                elif eye_yaw <= eye_yaw_min:
                    neck_yaw= eye_yaw - eye_yaw_min
                    eye_yaw = np.clip(eye_yaw, eye_yaw_min, eye_yaw_max)
                    neck_yaw= np.clip(neck_yaw, neck_yaw_min, neck_yaw_max)

                if eye_pitch >= eye_pitch_max :
                    neck_pitch= eye_pitch-eye_pitch_max
                    eye_pitch = np.clip(eye_pitch, eye_pitch_min, eye_pitch_max)
                    neck_pitch= np.clip(neck_pitch, neck_pitch_min, neck_pitch_max)

                elif eye_pitch <= eye_pitch_min:
                    neck_pitch= eye_pitch-eye_pitch_min
                    eye_pitch = np.clip(eye_pitch, eye_pitch_min, eye_pitch_max)
                    neck_pitch= np.clip(neck_pitch, neck_pitch_min, neck_pitch_max)

            
            p.setJointMotorControl2(self.robot_id, 0, p.POSITION_CONTROL, targetPosition=neck_yaw)
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=neck_pitch)
            p.setJointMotorControl2(self.robot_id, i-2, p.POSITION_CONTROL, targetPosition=eye_yaw)
            p.setJointMotorControl2(self.robot_id, i-1, p.POSITION_CONTROL, targetPosition=eye_pitch)
            


######################################################################
        
    def inverse_scale_action(self, joint_name, real_value):
        """
        Convert a real joint value to its normalized [0,1] representation.

        :param joint_name: Name of the joint.
        :param real_value: Real joint value.
        :return: Normalized value in [0,1] range.
        """
        min_val, max_val, _ = JOINT_LIMITS[joint_name]
        return (real_value - min_val) / (max_val - min_val)

    def scale_action(self, joint_name, normalized_value):
        """
        Scale an action from [0,1] range to its actual joint limits.
        """
        min_val, max_val, _ = JOINT_LIMITS[joint_name]
        return min_val + normalized_value * (max_val - min_val)

    def step(self, action=None, num_steps=10):
        """
        Perform a simulation step.

        :param action: Dictionary of joint_id to target position values in [0,1] range.
        :param num_steps: Number of simulation steps to run for each action.
        """
        self.step_count += 1

        if self.debug_tool:
            # Read slider values (normalized) and scale to real joint limits
            for joint_id, slider_id in self.joint_sliders.items():
                joint_info = p.getJointInfo(self.robot_id, joint_id)
                joint_name = joint_info[1].decode("utf-8")

                if joint_name in JOINT_LIMITS:
                    normalized_value = p.readUserDebugParameter(slider_id)
                    scaled_value = self.scale_action(joint_name, normalized_value)
                    p.setJointMotorControl2(
                        bodyUniqueId=self.robot_id,
                        jointIndex=joint_id,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=scaled_value
                    )
        else:
            # Apply external action commands if provided
            if action:
                for joint_id, normalized_value in action.items():
                    joint_info = p.getJointInfo(self.robot_id, joint_id)
                    joint_name = joint_info[1].decode("utf-8")

                    if joint_name in JOINT_LIMITS:
                        scaled_value = self.scale_action(joint_name, normalized_value)
                        p.setJointMotorControl2(
                            bodyUniqueId=self.robot_id,
                            jointIndex=joint_id,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=scaled_value
                        )

        # Run simulation for a fixed number of steps
        for _ in range(num_steps):
            self.eye_tracking()
            p.stepSimulation()
            time.sleep(self.time_step)
        
        # if self.step_count % self.eyetracking_update_interval == 0:
        #     self.eye_tracking()
        if self.usecamera==True:
            if self.step_count % self.camera_update_interval == 0:
                self.camera()


    def disconnect(self):
        """ Disconnects from the PyBullet simulation. """
        p.disconnect()


if __name__ == "__main__":
    # Example usage
    #the usefixedbase is used to set the added square object to be fixed.
    env = FaceRobotEnv(use_gui=True, debug_tool=False, usefixedbase=False ,object_location=[1.2, -0.45, 0.2],usecamera=False) #edit the object location here

    try:
        for _ in range(1000):  # Run simulation for 1000 steps
            env.step(num_steps=10)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    # Disconnect from PyBullet
    env.disconnect()
    cv2.destroyAllWindows()
