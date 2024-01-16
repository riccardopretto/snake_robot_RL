import gymnasium as gym
import mujoco
from sb3_contrib import TRPO
import numpy as np
import os
import matplotlib.pyplot as plt
from array import *
import pandas as pd
import shutil
import sys
import snake_v14
from PIL import Image as im
from tempfile import TemporaryDirectory
import ffmpeg
from scipy.integrate import simpson
import seaborn as sns


# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1080, 1080))
# display.start()


#-----------------------------------------------------------------------------------------------------------------
#█████████████████████████████████████████████████████████████████████████████████████████████████████████████████
#-----------------------------------------------------------------------------------------------------------------

snake_ver = 14

filename = "trained_model"
param = "_in_2_2"
save_dir = f"{filename}_dir{param}"
#filename = sys.argv[0]

RANDOM_FRICTION = True

iteration_range = 500

VIDEO_RENDER = True 

SAVE_PLOTS = True

#-----------------------------------------------------------------------------------------------------------------
#█████████████████████████████████████████████████████████████████████████████████████████████████████████████████
#-----------------------------------------------------------------------------------------------------------------

#variable declaration
info_num_list = []
info_num = []
motion_data_array_info = []
motion_data_info = np.zeros(10)
motion_data_obs = np.zeros(22)
motion_data_action =np.zeros(7)
reward_arr = np.zeros(1)
energy_per_motor_vec =np.zeros(7)
energy_per_motor=[]
goal_reached = False
i_goal_reached = 0

#---------plots
obs_plot_titles = ["x", "y", "$\\theta_0$", "$\\theta_1$", "$\\theta_2$", "$\\theta_3$", "$\\theta_4$", "$\\theta_5$", "$\\theta_6$", "$\\theta_7$",
                   "\dot x", "\dot y", "$\dot \\theta_0$", "$\dot \\theta_1$", "$\dot \\theta_2$", "$\dot \\theta_3$", "$\dot \\theta_4$", "$\dot \\theta_5$", "$\dot \\theta_6$", "$\dot \\theta_7$",
                   "x_distance_to_goal", "y_distance_to_goal"]
info_plot_titles = ["goal reward", "control reward", "velocity tracking reward", "total_reward", "distance_from_goal", "x_position", "y_position", "x_velocity", "y_velocity", "velocity"]


model_path = f"{filename}"

tmp_dir_path = "000_frames"
if not os.path.exists(tmp_dir_path):
    os.makedirs(tmp_dir_path)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)




#-----------------create the enviroment
env = gym.make(f"Snake-v{snake_ver}", render_mode = "rgb_array", width = 1080, height = 720)
env.reset()

model = TRPO.load(model_path, env=env)

vec_env = model.get_env()
obs = vec_env.reset()

goal_pos = snake_v14.SnakeEnv.get_goal_pos(env)

frame_counter = 0

for i in range(iteration_range):

    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)



    if VIDEO_RENDER:
        frame_arr = vec_env.render(mode="rgb_array") 
        frame = im.fromarray(frame_arr) 
        frame_counter = frame_counter + 1 
        frame.save(f"{tmp_dir_path}/frame_{frame_counter:06d}.png") 
        print("Saved video frame num: ",frame_counter)


    
    #-------------------------------data OBS
    for j in range(len(obs)):
        motion_data_array_obs = obs[j]
        motion_data_obs = np.vstack([motion_data_obs, motion_data_array_obs])


    #-------------------------------PLOT INFO
    for key, value in info[0].items():
        if isinstance(value, (int, float)):
            info_num.append(value)
    info_num.pop()  # Removes the False from the last element
    motion_data_array_info = info_num
    motion_data_info = np.vstack([motion_data_info, motion_data_array_info])
    info_num =[]
    


    #-------------------------------PLOT ACTIONS
    for j in range(len(action)):
        motion_data_array_action = action[j] * 1.4
        motion_data_action = np.vstack([motion_data_action, motion_data_array_action])
    

    #-------------------------------PLOT REWARD
    reward_arr = np.vstack([reward_arr, reward])
    

    # VecEnv resets automatically
    if done:
        obs = env.reset()
    #end for




#delete the first row, because it's only 0s
motion_data_obs = motion_data_obs[1:1000,:]
motion_data_info = motion_data_info[1:1000,:]
motion_data_action = motion_data_action[1:1000,:]
reward_arr = reward_arr[1:1000,:]


#----------energy
pos_motor = motion_data_obs[:,2:10]
energy_per_motor = [abs(simpson(y=motion_data_action[:,mot_idx], x=pos_motor[:,mot_idx], even="avg")) for mot_idx in range(7)]
energy = np.sum(energy_per_motor)


#---------goal reached time
if i_goal_reached == 0:
    goal_reached_time = "goal never reached"
else:
    goal_reached_time = i_goal_reached * 0.04


#-----------------------------------------------------------save in CSV FILE
#----------OBS
motion_data_obs_df = pd.DataFrame(motion_data_obs)

#-------INFO
motion_data_info_df = pd.DataFrame(motion_data_info)

#----------ACTION
motion_data_action_df = pd.DataFrame(motion_data_action)

#----------ENERGY
energy_df = pd.DataFrame(np.array([energy]))

#----------GOAL REACHED
goal_reached_time_df = pd.DataFrame(np.array([goal_reached_time]))

#save csv files
motion_data_obs_df.to_csv(f'{save_dir}/obs.csv', sep=',', header=False, float_format='%.5f', index=False)
motion_data_info_df.to_csv(f'{save_dir}/info.csv', sep=',', header=False, float_format='%.5f', index=False)
motion_data_action_df.to_csv(f'{save_dir}/action.csv', sep=',', header=False, float_format='%.5f', index=False)
energy_df.to_csv(f'{save_dir}/energy.csv', sep=',', header=False, float_format='%.5f', index=False)
goal_reached_time_df.to_csv(f'{save_dir}/goal_reached.csv', sep=',', header=False, float_format='%.5f', index=False)

if SAVE_PLOTS:
    #------------------PLOT-------------------
    x = np.arange(0, iteration_range)
    x_sec = x*0.04      #transform x in seconds

    #---------PLOT OBS POS-----------------
    plt.figure(figsize=(15,10))
    plt.suptitle(f'OBS positions', fontsize=28)
    for i in range(10): #0-9 only pos
        if i != 0 and i != 1:
            motion_data_obs[:,i] = motion_data_obs[:,i]*180/3.14159 #transform to rad in degrees
        plt.subplot(5,2,i+1)
        plt.tight_layout(pad=2.0)
        plt.grid()
        plt.plot(x_sec, motion_data_obs[:,i])
        plt.title(obs_plot_titles[i])
        plt.xlabel('time [s]')
        if i<=1:
            plt.ylabel('m')
        if i>=2 and i<=9:
            plt.ylabel('deg')
    plt.savefig(f'{save_dir}/obs_pos.png',bbox_inches="tight")
    plt.savefig(f'{save_dir}/obs_pos.pdf',bbox_inches="tight")


    #---------PLOT OBS VEL-----------------
    plt.figure(figsize=(15,10))
    plt.suptitle(f'OBS velocities', fontsize=28)
    for i in range(10): #10-19 only pos
        if i != 0 and i != 1:
            motion_data_obs[:,i+10] = motion_data_obs[:,i+10]*180/3.14159 #transform to rad in degrees
        plt.subplot(5,2,i+1)
        plt.tight_layout(pad=2.0)
        plt.grid()
        plt.plot(x_sec, motion_data_obs[:,i+10])
        plt.title(obs_plot_titles[i+10])
        plt.xlabel('time [s]')
        if i<=1:
            plt.ylabel('m/s')
        if i>=2 and i<=9:
            plt.ylabel('deg/s')
    plt.savefig(f'{save_dir}/obs_vel.png',bbox_inches="tight")
    plt.savefig(f'{save_dir}/obs_vel.pdf',bbox_inches="tight")

    #---------PLOT OBS DISTANCE-----------------
    plt.figure(figsize=(10,5))
    plt.suptitle(f'OBS goal distances', fontsize=28)
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.plot(x_sec, motion_data_obs[:,i+20])
        plt.title(obs_plot_titles[i+20])
        plt.xlabel('time [s]')
        plt.ylabel('m')
        plt.tight_layout(pad=2.0)
        plt.grid()
    plt.savefig(f'{save_dir}/obs_dist.png',bbox_inches="tight")
    plt.savefig(f'{save_dir}/obs_dist.pdf',bbox_inches="tight")



    #----------PLOT INFO-----------
    plt.figure(figsize=(15,10))
    plt.suptitle(f'INFO', fontsize=28)
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.plot(x_sec, motion_data_info[:,i])
        plt.title(info_plot_titles[i])
        plt.xlabel('time [s]')
        plt.tight_layout(pad=2.0)
        plt.grid()
    plt.savefig(f'{save_dir}/info.png',bbox_inches="tight")
    plt.savefig(f'{save_dir}/info.pdf',bbox_inches="tight")


    #----------PLOT 2D with goal-----------
    goal_circle = plt.Circle((goal_pos[0], goal_pos[1]), 0.01, color='r')
    fig,ax = plt.subplots(figsize=(10,10))
    plt.plot(motion_data_info[:,5], motion_data_info[:,6])
    ax.add_patch(goal_circle)
    plt.xlabel('x position [m]')
    plt.ylabel('y position [m]')
    plt.title(f'2D_movement')
    plt.grid()
    plt.savefig(f'{save_dir}/2D_movement.png',bbox_inches="tight")
    plt.savefig(f'{save_dir}/2D_movement.pdf',bbox_inches="tight")


    #----------PLOT ACTION-----------
    plt.figure(figsize=(15,10))
    plt.suptitle(f'ACTION', fontsize=24)
    for i in range(7):
        plt.subplot(4, 2, i+1)
        plt.plot(x_sec, motion_data_action[:,i])
        plt.xlabel('time [s]')
        plt.ylabel('Torque [Nm]')
        plt.title(f"motor_{i+1}")
        plt.tight_layout(pad=2.0)
        plt.grid()
    plt.savefig(f'{save_dir}/action.png',bbox_inches="tight")
    plt.savefig(f'{save_dir}/action.pdf',bbox_inches="tight")


    #goal distribution
    headings = ["x", "y"]
    dic = {x:[] for x in headings}
    goal_used_pos = np.loadtxt(f"goal_used.csv",delimiter=",")
    goal_used_pos=np.delete(goal_used_pos,0,0)
    dic["x"].extend(goal_used_pos[:,0])
    dic["y"].extend(goal_used_pos[:,1])
    df = pd.DataFrame(dic)
    # print(df)

    plt.figure(figsize=(10,10))
    plt.suptitle(f'Goal distribution during training', fontsize=22)
    plt.hexbin(goal_used_pos[:,0], goal_used_pos[:,1],cmap="hot")
    plt.grid()
    plt.savefig(f'{save_dir}/goals_distrib.png',bbox_inches="tight")
    plt.savefig(f'{save_dir}/goals_distrib.pdf',bbox_inches="tight")
    plt.clf()



if VIDEO_RENDER:
    #video_rendering
    (
    ffmpeg
    .input(f"{tmp_dir_path}/frame_*.png", pattern_type='glob', framerate=25)
    .output(f'{filename}{param}.mp4', vcodec="libx264")
    .run()
    )
    #delete frames folder
    frame_counter = 1
    while FileNotFoundError:
        os.remove(f"{tmp_dir_path}/frame_{frame_counter:06d}.png")
        frame_counter = frame_counter + 1
    print("temp folder deleted")