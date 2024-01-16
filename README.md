# snake_robot_RL

Use train.py and load_trained_model.py for respectively train and load the trained model. 
In load_trained_model.py you can even save the video and the plots of obs, action, rewards etc.

The xml file is the simulation file for mujoco. Use mujoco-2.3.7 ! 

In snake.py are described the snake's obs, action and what to do every time the env is resetted dring training. There you can either choose to use random or fixed friction and goal position.

parallel_env.py is just used for the env parallelization during training.
