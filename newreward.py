# A reward function that minimizes the human input that might manipulate the agent's behavior.
import math
import numpy as np

try:
    if 'RACE_LINE' not in globals():
        global RACE_LINE
        RACE_LINE = np.load('/opt/ml/input/config/custom_files/my_race_line.npy')
except Exception as e:
    print(f"Error loading custom waypoints: {e}")
    RACE_LINE = None

# Edit: 

def reward_function(params):
    """
    Compute a reward for AWS DeepRacer given telemetry parameters.
    """

    # ---- CONSTANTS ----
    MAX_REWARD = 10
    MIN_REWARD = 1e-3
    ABS_STEERING_THRESHOLD = 30
    DIRECTION_THRESHOLD = 90

    # ---- INPUT PARAMETERS ----
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    speed = params['speed']
    steering_angle = params['steering_angle']
    progress = params['progress']
    steps = params['steps']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    is_crashed = params['is_crashed']
    is_offtrack = params['is_offtrack']
    is_reversed = params['is_reversed']
    track_length = params['track_length']
    x = params['x']
    y = params['y']
    is_left_of_center = params['is_left_of_center']

    # Initialize reward (WE MIGHT NOT HAVE TO DO THIS)
    reward = 1.0
    reward_ontrack = 1.0
    reward_steering = 1.0
    reward_step = 1.0
    reward_throttle = 1.0
    reward_speed = 1.0
    reward_raceline = 1.0
    reward_proximity = 1.0
    reward_waypoint = 1.0

    # ---- TIME TRIAL FUNCTIONS ----
    def on_track_reward(current_reward):    # reward if the wheels stay inside the two borders of the track
        if all_wheels_on_track and (0.5 * track_width - distance_from_center) >= 0.1:
            current_reward = MAX_REWARD
        else:
            current_reward *= MIN_REWARD        # heavily penalize if even 1 wheel is off track
        if is_offtrack or is_crashed or is_reversed:
            current_reward *= 0.1
        return current_reward

    def steering_reward(current_reward):    # reward for steering in the direction of the track
        track_direction = math.degrees(
            math.atan2(wp2[1] - wp0[1], wp2[0] - wp0[0])
        )
        direction_diff = abs(track_direction - heading)
        if abs(steering_angle - direction_diff) < 10.0:
            current_reward *= 2.0  # reward minimal steering

        if r_value > 0.8 and abs(steering_angle) < 5.0:
            current_reward *= 2.0  # reward minimal steering on straight sections
        
        if abs(steering_angle) > 20:
            current_reward *= 0.5  # penalize excessive steering
        
        return current_reward

    def step_reward(current_reward):        #reward for completing the track in fewer steps
        if steps > 0:
            current_reward *= progress / steps
        if progress == 100:
            current_reward *= 1000 / steps
        return current_reward

    def throttle_reward(current_reward):    # reward for maintaining a higher speed, but not too high in curves
        if abs(steering_angle) > 10 and speed <= 2.5 - (0.05 * abs(steering_angle)):
            current_reward *= 2.0
        return current_reward


    # maybe have a separate reward for straight vs curve sections, so that its weights can be adjusted individually.

    def speed_reward(current_reward):       # reward for adjusting speed according to the straightness of the track
        if r_value < 0.5 and speed > 1.5:
            current_reward *= 0.5  # slow down for curvier sections
        elif r_value < 0.5 and speed <= 1.5:
            current_reward *= 2.0  # reward for slowing down for curvier sections
        # speed for straight sections
        elif r_value >= 0.8 and speed < 3.0:
            current_reward *= 0.5
        elif r_value >= 0.8 and speed >= 3.0:
            current_reward *= 2.0  # reward for speeding up on straighter sections

        return current_reward

    def waypoint_reward(current_reward):

        if RACE_LINE is None:
            return 1e-3 # Fail fast if the file didn't load

        # --- 2. Get car's position ---
        car_pos = np.array([params['x'], params['y']])

        # --- 3. Find the distance to *your* race line ---
        # This finds the shortest distance from the car to any point on your custom line
        distances = np.linalg.norm(RACE_LINE - car_pos, axis=1)
        min_distance_to_race_line = np.min(distances)

        # --- 4. Reward proximity to *your* line ---
        # It rewards being close to the RACE_LINE, while the car
        # *sees* the simulation with the *correct* visual centerline.
        
        # A "corridor" width to aim for (e.g., 0.15 meters)
        corridor_width = 0.15 
        sharpness = 3 # A sharpness of 5 is very high, 3 is safer

        reward = 2 * np.exp(-abs(min_distance_to_race_line / corridor_width) * sharpness)
        
        return current_reward

    """def brake_reward(current_reward):        #reward for braking before curves
        # compute the path direction of waypoints 2 and 3 ahead
        path_direction = math.degrees(
            math.atan2(wp3[1] - wp1[1], wp3[0] - wp1[0]))
        direction_diff = abs(path_direction - heading)
        if direction_diff > DIRECTION_THRESHOLD and speed > 2.0:
            current_reward *= 0.5  # penalize not braking before curves
        elif direction_diff > DIRECTION_THRESHOLD and speed <= 2.0:
            current_reward *= 2.0  # reward braking before curves

        return current_reward"""


# The raceline reward function is no longer needed, since the waypoints already represent the optimal racing line.

    """def raceline_reward(current_reward):    # reward for following the optimal racing line through curves
        # compute track direction using four waypoints ahead/behind
        track_direction = math.degrees(
            math.atan2(wp5[1] - wp4[1], wp5[0] - wp4[0])
        )

        # right curve
        if (
            (track_direction - heading) < -20
            and not is_left_of_center
            and 0.1 < distance_from_center < (track_width / 2.0)
        ):
            current_reward *= 1.3

        # left curve
        elif (
            (track_direction - heading) > 20
            and is_left_of_center
            and 0.1 < distance_from_center < (track_width / 2.0)
        ):
            current_reward *= 1.3

        # smooth operation
        if abs(track_direction - heading) < 15.0 and abs(steering_angle) < 10.0:
            current_reward *= 1.2

        # right extreme curve
        if ((track_direction - heading) < -140.0
            and is_left_of_center
            and 0.1 < distance_from_center < (track_width / 2.0)
        ):
            current_reward *= 1.3

        # left extreme curve
        elif ((track_direction - heading) > 140.0
            and not is_left_of_center
            and 0.1 < distance_from_center < (track_width / 2.0)
        ):
            current_reward *= 1.3

        return current_reward"""

    # ---- INITIALIZE REWARD DEPENDING ON WHETHER ON TRACK ----
    reward = on_track_reward(reward)

    # ---- COMBINE REWARD COMPONENTS ----
    reward_ontrack = on_track_reward(reward)    
    reward_steering = steering_reward(reward)  
    reward_step = step_reward(reward)
    reward_throttle = throttle_reward(reward)  
    reward_speed = speed_reward(reward)           
    reward_raceline = raceline_reward(reward)  
    reward_waypoint = waypoint_reward(reward)
    # reward_proximity = proximity_reward(reward_proximity)

    # specialized straight
    if r_value > 0.8:
        reward = (
            (0.8)*reward_steering + 
            (0.8)*reward_step + 
            (0.5)*reward_throttle + 
            (0.8)*reward_speed + 
            (0.5)*reward_raceline +
            (1.0)*reward_waypoint
            # + (0.5)*reward_proximity
            # if the throttle reward becomes less significant, the car will start drifting again
        )
        
    # specialized curve
    elif r_value < 0.5:
        reward = (
            (0.5)*reward_steering + 
            (0.5)*reward_step + 
            (1.0)*reward_throttle + 
            (0.5)*reward_speed + 
            (1.0)*reward_raceline +
            (1.0)*reward_waypoint
            # + (0.5)*reward_proximity
            # if the throttle reward becomes less significant, the car will start drifting again
        )
    # transition
    else:
        reward = (
            (0.5)*reward_steering + 
            (0.7)*reward_step + 
            (0.8)*reward_throttle + 
            (0.8)*reward_speed + 
            (0.7)*reward_raceline +
            (1.0)*reward_waypoint
            # + (0.5)*reward_proximity
            # if the throttle reward becomes less significant, the car will start drifting again
        )
    return reward