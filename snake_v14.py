import numpy as np
import mujoco as mujoco
import gymnasium
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from os import path
from os import listdir
import sys
import pandas as pd
import random


DEFAULT_CAMERA_CONFIG = {}

gymnasium.envs.register(
    id="Snake-v14",
    entry_point=f"{__name__}:SnakeEnv",
    max_episode_steps=750,  #duration of an episode - 1000 stands for 40 second at 25fps
    reward_threshold=1000,
)

xml_filename = "snake_v14.xml"                  #name of the model file here
directory, filename = path.split(__file__)
xml_file_path = path.join(directory, xml_filename)


class SnakeEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        xml_file=xml_file_path,
        forward_reward_weight=1.0,
        reset_noise_scale=0.01,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            forward_reward_weight,
            reset_noise_scale,
            **kwargs,
        )

        self.max_vel = 0.3      #m/s
        self.max_ctrl_cost = 1.4*7
        #---------------------------------------------------------------------------- WEIGHT
        self._forward_reward_weight = forward_reward_weight #not used
        self._ctrl_cost_weight = 0.02           #control cost weight
        self.goal_reward_weight = 0.98          #goal reward weight
        self.vel_tracking_weight = 0.05         #in case you want to do velocity tracking too
        #self._ctrl_cost_weight = ctrl_cost_weight
        self.x_vel_tracking_weight = 0.5        #if you want to track x and y indipendently
        self.y_vel_tracking_weight = 0.3        #if you want to track x and y indipendently
        #---------------------------------------------------------------------------- 
        
        self.goal_used_for_training = [0,0]

        self._prev_action = 0
        self._prev_x_pos = 0

        self.goal_pos = [0,0]
        self.goal_bound = 0.01        #goal boundaries
        self.goal_distance = 0
        self.goal_reached = False
        self.initial_goal_distance = 0

        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float64     #it stands for: [x,y,z,theta0,t1,t2,t3,t4,t5,t6,t7,d_x,d_y,d_z,d_t1, d_t2, ... , x_goal-current_x_pos, y_goal-current_y_pos] positions, velocities and goal distance
        )

        MujocoEnv.__init__(
            self, xml_file, 20, observation_space=observation_space, **kwargs
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.square(np.sum(np.square(action)))/self.max_ctrl_cost        #normalized
        return control_cost

    def step(self, action):
        action = action * 1.4         #---------------------------------action[-1,1] * gain so the max torque is +-1.4Nm
        xy_position_before = self.data.qpos[0:2].copy()

        self.do_simulation(action, self.frame_skip)                     #here it does the simualtion

        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)

        #get the new step observation
        observation = self._get_obs()

        #velocity
        tot_velocity = np.sum(np.square(xy_velocity))
        vel_tracking_reward = abs(self.max_vel - tot_velocity)/self.max_vel*self.vel_tracking_weight       #normalized

        x_vel_tracking_reward = abs(self.max_vel - x_velocity)/self.max_vel*self.x_vel_tracking_weight       #normalized
        y_vel_tracking_reward = abs(0 - y_velocity)/self.max_vel*self.y_vel_tracking_weight       #normalized

    

        #goal distance
        x_goal_dist = abs(self.goal_pos[0]-observation[0])   #should go before get obs
        y_goal_dist = abs(self.goal_pos[1]-observation[1])
        self.goal_distance = np.sqrt(np.square(x_goal_dist) + np.square(y_goal_dist))

        goal_reward = self.goal_reward(goal_distance = self.goal_distance)

        if self.goal_distance < self.goal_bound:
            self.goal_reached = True
        
        #------------------REWARD-------------------
        if not self.goal_reached:
            reward = goal_reward - ctrl_cost 
        else:
            reward = goal_reward
        #------------------REWARD-------------------
        
        self._prev_x_pos = observation[0]
        self._prev_action = action

        info = {
            "goal_reward": goal_reward,
            "reward_ctrl": -ctrl_cost,
            "velocity_tracking_reward": vel_tracking_reward,
            "total_reward": reward,
            "distance_from_goal": self.goal_distance,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "tot_velocity": tot_velocity,
        }


        if self.render_mode == "human":
            self.render()

        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        position = np.delete(position,2)    #delete z obs
        velocity = self.data.qvel.flat.copy()
        velocity = np.delete(velocity,2)    #delete z_dot obs
        goal_dist_state = [position[0]-self.goal_pos[0], position[1]-self.goal_pos[1]]   #add 2 state
        observation = np.concatenate([position, velocity, goal_dist_state]).ravel()

        return observation

    def reset_model(self):      #every time the env resets, here there are the inizialized variables
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        #GOAL
        self.goal_pos = self.set_rndm_goal(min_dist = 1.50, max_dist = 2.10)  #sphere goal xy position, random goal
        #test
        self.goal_reached = False
        # self.goal_pos = [2,2]           #---------------------------fixed goal, you can even use a random goal
    

        #--------------------------------in case you want to keep track of the goals used
        # self.goal_used_for_training = np.vstack((self.goal_used_for_training,self.goal_pos))
        # goal_df = pd.DataFrame(self.goal_used_for_training)
        # goal_df.to_csv("goal_used.csv", sep=',', header=False, float_format='%.2f', index=False)


        initial_goal_distance_x = self.goal_pos[0] - qpos[0]
        initial_goal_distance_y = self.goal_pos[1] - qpos[1]
        self.initial_goal_distance = np.sqrt(np.square(initial_goal_distance_x) + np.square(initial_goal_distance_y))


        #FRICTION
        #place her friction randomness
        # f1_rndm = round(random.uniform(0.10, 1.00), 2)
        # f2_rndm = round(random.uniform(0.10, 1.00), 2) 
        # f3_rndm = round(random.uniform(0.10, 0.50), 2)
        f1_rndm = 0.3
        f2_rndm = 0.7
        f3_rndm = 0.05
        friction_values = [f1_rndm, f2_rndm, f3_rndm, 0.0001, 0.0001]
        #set friction
        for idx in range(8):
            self.set_pair_friction(geom_name1=f"caps{idx+1}", geom_name2="floor", new_pair_friction=friction_values)


        self.set_state(qpos, qvel)

        observation = self._get_obs()
        #print("Env resetted")
        return observation
    
    def set_rndm_goal(self, min_dist, max_dist):
        """Set a rndm position goal between min e max distance.

        :min_dist = minimum distance in x and y, ex min_value = 1 then x and y are more than 1
        :max_dist = maximum distance in x and y, ex max_value = 2 then x and y are less than 2

        """
        #set random goal sphere position
        x_goal_rndm = 0.00
        y_goal_rndm = 0.00
        #while goal is between x and y [-min_dist, min_dist]
        while x_goal_rndm < min_dist and x_goal_rndm > -min_dist:
            x_goal_rndm = round(random.uniform(-max_dist, max_dist), 2)
        while y_goal_rndm < min_dist and y_goal_rndm > -min_dist:
            y_goal_rndm = round(random.uniform(-max_dist, max_dist), 2)
        goal_pos_rndm = [abs(x_goal_rndm),y_goal_rndm]                  #only x positive

        return goal_pos_rndm
    
    def get_goal_pos(self):
        goal_pos = self.goal_pos
        return goal_pos

    def goal_reward(self, goal_distance, w=1, v=1, alpha=1e-4):
        goal_reward = -w*goal_distance**2 -v*np.log(goal_distance**2 + alpha)
        min_goal_reward = -w*self.initial_goal_distance**2 -v*np.log(self.initial_goal_distance**2 + alpha)
        max_goal_reward = -w*0**2 -v*np.log(0**2 + alpha)
        normalized_goal_reward = (goal_reward - min_goal_reward)/(abs(max_goal_reward)+abs(min_goal_reward))
        return normalized_goal_reward*self.goal_reward_weight

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
        

    #----------------------------VARYING FRICTION---------------------------------------


    def get_id(self, name, obj_type):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    

    def get_contact_pair(self, geom_name1, geom_name2):
        """Gets the pair ID of two objects that are in contact. 
        If the objects are not in contact it wil raise a key error.

        :param geom_name1: The geom name of the first object in enviroment
        :type geom_name1: str
        :param geom_name2: The geom name of the second object in enviroment
        :type geom_name2: str
        :return: The contact pair ID
        :rtype: int
        """
        # Find the proper geom ids
        geom_id1 = self.get_id(geom_name1, 'geom')
        geom_id2 = self.get_id(geom_name2, 'geom')

        # Find the right pair id
        pair_geom1 = self.model.pair_geom1
        pair_geom2 = self.model.pair_geom2
        pair_id = None
        for i, (g1, g2) in enumerate(zip(pair_geom1, pair_geom2)):
            if g1 == geom_id1 and g2 == geom_id2 \
                or g2 == geom_id1 and g1 == geom_id2:
                pair_id = i
                #print("pair_id n=", pair_id)  #i should get 9 values
                break 
        if pair_id is None: 
            raise KeyError("No contact between %s and %s defined."
                            % (geom_name1, geom_name2))
        return pair_id

    def get_pair_solref(self, geom_name1, geom_name2):
        """Gets the solver parameters between two objects in the enviroment.
        The solref represents the stiffness and damping between two object
        (mass-spring-damper system). See Mujoco documentation.

        :param geom_name1: The geom name of the first object in enviroment
        :type geom_name1: str
        :param geom_name2: The geom name of the second object in enviroment
        :type geom_name2: str
        :return: The solref between two objects
        :rtype: ???
        """
        pair_id = self.get_contact_pair(geom_name1, geom_name2)
        pair_solref = self.model.pair_solref[pair_id]
        return pair_solref

    def get_pair_friction(self, geom_name1, geom_name2):
        """Gets the friction between two objects in the 
        enviroment with geom_names specified.

        :param geom_name1: The geom name of the first object in enviroment
        :type geom_name1: str
        :param geom_name2: The geom name of the second object in enviroment
        :type geom_name2: str
        :return: The friction between two objects
        :rtype: ???
        """
        pair_id = self.get_contact_pair(geom_name1, geom_name2)
        pair_friction = self.model.pair_friction[pair_id]
        return pair_friction
    
    #set pair friction
    def set_pair_friction(self, geom_name1, geom_name2, new_pair_friction):
        """    
        :description: Sets the friction between 2 geoms
        :param new_pair_friction: New friction value. Has to be an array of 5 elements
        """
        pair_id = self.get_contact_pair(geom_name1, geom_name2)
        self.model.pair_friction[pair_id] = new_pair_friction
        return self.model.pair_friction[pair_id]
    


    @property
    def skin_friction(self):
        return self.get_pair_friction("seg1_body", "floor")     #i have from seg1 to seg9!!!!!

    @skin_friction.setter
    def skin_friction(self, value):
        """    
        :description: Sets the friction between the puck and the sliding surface
        :param value: New friction value. Can either be an array of 2 floats
            (to set the linear friction) or an array of 5 float (to set the
            torsional and rotational friction values as well)
        :raises ValueError: if the dim of ``value`` is other than 2 or 5
        """
        pair_fric = self.get_pair_friction("seg1_body", "floor")  #i have from seg1 to seg9!!!!!
        if value.shape[0] == 2:
            # Only set linear friction
            pair_fric[:2] = value
        elif value.shape[0] == 3:
            # linear friction + torsional
            pair_fric[:3] = value
        elif value.shape[0] == 5:
            # Set all 5 friction components
            pair_fric[:] = value
        else:  
            raise ValueError("Friction should be a vector or 2 or 5 elements.")
