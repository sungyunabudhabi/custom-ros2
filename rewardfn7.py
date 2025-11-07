import math
import numpy as np

# Edit: remove speed and brake rewards, and add raceline waypoint proximity and alignment reward.

RACE_LINE = np.array([
    [2.8873885499889385, 0.726467741007691],
    [3.1675912209711705, 0.704786488311369],
    [3.4551731738475713, 0.6921786257120108],
    [3.7532515756822287, 0.6858100463187193],
    [4.072814338243193, 0.6836081931173711],
    [4.500002230529587, 0.6837609167129147],
    [4.549995073956144, 0.6837787896136626],
    [5.117381147897781, 0.6908041075226552],
    [5.447982562982464, 0.7112322029646044],
    [5.711265580422473, 0.7422346953355425],
    [5.941372105449603, 0.7849646168697144],
    [6.14912709690292, 0.8407803487746184],
    [6.336758932063642, 0.910667356962588],
    [6.503516690922521, 0.9948399398813299],
    [6.647625884029711, 1.0933636666158342],
    [6.767148492564518, 1.2064015782188044],
    [6.857904170606057, 1.3350866866561408],
    [6.921937617475411, 1.4764660946579922],
    [6.960268235163922, 1.6279734619836947],
    [6.9668995801856575, 1.7888071991434118],
    [6.929767421700819, 1.9551543405695744],
    [6.853796172049812, 2.119102709675103],
    [6.7269327251774005, 2.268416334375874],
    [6.565827306421308, 2.3979064968172388],
    [6.380755123221848, 2.5063265223251174],
    [6.180371705554723, 2.5960264990184765],
    [5.97126498649426, 2.6720718677404474],
    [5.75829177463112, 2.741103010083892],
    [5.558411769338754, 2.810132384390418],
    [5.360049465761351, 2.88360578275889],
    [5.163331306801121, 2.962188029071304],
    [4.968449030193712, 3.0468263416666996],
    [4.775520319459698, 3.138325431489057],
    [4.584624396203516, 3.237452798325894],
    [4.395624809243852, 3.344197010326614],
    [4.208250346496813, 3.4578934289044962],
    [4.022165217846894, 3.577403746276934],
    [3.8371280683829303, 3.7018419231567705],
    [3.681861411012094, 3.809703887563246],
    [3.5252922717094037, 3.9117983675963584],
    [3.3667407341899676, 4.006064126580926],
    [3.2053248589385617, 4.090414741034139],
    [3.0401251953842676, 4.163356432852998],
    [2.8702442130275108, 4.22393077448964],
    [2.6948633514524545, 4.271622788605077],
    [2.5131932089848545, 4.306023645894755],
    [2.324525679916916, 4.326723820809339],
    [2.1269630912619206, 4.330802975266311],
    [1.9181050793836385, 4.3138121184730105],
    [1.6947191320004, 4.267408676972845],
    [1.4541627337365117, 4.174008490919465],
    [1.2111900454555669, 4.006532228942984],
    [1.0192295285380961, 3.744022018008346],
    [0.9222054925325214, 3.4205054375538393],
    [0.8892660439411404, 3.1044388864862364],
    [0.896007467184927, 2.8207603596691184],
    [0.9240494312211823, 2.5628118514410283],
    [0.9660525314669286, 2.324603051064647],
    [1.0180283266844032, 2.1122854381532052],
    [1.08079016698792, 1.915129811182102],
    [1.1551369760045151, 1.7310757057890727],
    [1.2416231723876427, 1.5601480656474387],
    [1.341129981140405, 1.4032388427610951],
    [1.4547258930465157, 1.2610931996668353],
    [1.5865309544578126, 1.1364118322222136],
    [1.7447260773737912, 1.0322868812760757],
    [1.9265552947643938, 0.9430548066740163],
    [2.1328222834811292, 0.8677942536468937],
    [2.3641125208150084, 0.8067988687678815],
    [2.617512764005648, 0.759921447767141],
    [2.8873885499889385, 0.726467741007691]
], dtype=float)

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
    objects_distance = params['objects_distance']
    objects_speed = params['objects_speed']
    objects_heading = params['objects_heading']
    objects_left_of_center = params['objects_left_of_center']
    objects_location = params['objects_location']
    closest_objects = params['closest_objects']
    x = params['x']
    y = params['y']
    is_left_of_center = params['is_left_of_center']

    # Pre-compute some useful values
    wp_len = len(waypoints)
    idx0 = (closest_waypoints[0])
    idx1 = (closest_waypoints[1])
    idx2 = (closest_waypoints[1] + 1) % wp_len
    idx3 = (closest_waypoints[1] + 2) % wp_len
    idx4 = (closest_waypoints[1] + 3) % wp_len
    idx5 = (closest_waypoints[1] + 4) % wp_len
    wp0 = waypoints[idx0]
    wp1 = waypoints[idx1]
    wp2 = waypoints[idx2]
    wp3 = waypoints[idx3]
    wp4 = waypoints[idx4]
    wp5 = waypoints[idx5]

    # compute the straightness of the 4 closest waypoints
    # maybe use earlier waypoints because the car is braking too early
    x = np.array([wp1[0], wp2[0], wp3[0], wp4[0]])
    y = np.array([wp1[1], wp2[1], wp3[1], wp4[1]])

    r_matrix = np.corrcoef(x, y)
    r_value = abs(r_matrix[0, 1])

    if np.isnan(r_value):
        r_value = 1.0
    # r_value varies between 0 and 1, with 1 being a perfect straight line
    # we can use this to vary the speed reward
    # more reward for higher r_value (straighter section of track)
    # less reward for lower r_value (curvier section of track)

    # Initialize reward (WE MIGHT NOT HAVE TO DO THIS)
    reward = 1.0
    reward_ontrack = 1.0
    reward_steering = 1.0
    reward_step = 1.0
    reward_throttle = 1.0
    reward_speed = 1.0
    reward_raceline = 1.0
    reward_proximity = 1.0

    # ---- TIME TRIAL FUNCTIONS ----
    if is_offtrack or is_crashed or is_reversed:
        return MIN_REWARD

    def steering_reward(current_reward):    # reward for steering in the direction of the track
        track_direction = math.degrees(
            math.atan2(wp2[1] - wp1[1], wp2[0] - wp1[0])
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
            pps = progress / float(steps)
            target_pps = 0.36
            current_reward = max(0.5, min(pps / target_pps, 2.0))
        else:
            current_reward = 1.0
        return current_reward

    def throttle_reward(current_reward):    # reward for maintaining a higher speed, but not too high in curves
        if abs(steering_angle) > 10 and speed <= 2.5 - (0.05 * abs(steering_angle)):
            current_reward *= 2.0
        return current_reward

    def raceline_reward(current_reward):
        car_pos = np.array([params['x'], params['y']])

        # distance to race line
        distances = np.linalg.norm(RACE_LINE - car_pos, axis=1)
        min_distance = float(np.min(distances))

        corridor_width = 0.15
        sharpness = 3.0

        proximity_reward = np.exp(-(min_distance / corridor_width) * sharpness)

        idx = int(np.argmin(distances))
        if idx < len(RACE_LINE) - 1:
            next_point = RACE_LINE[idx + 1]
        else:
            next_point = RACE_LINE[idx - 1]

        race_direction = math.degrees(
            math.atan2(next_point[1] - RACE_LINE[idx][1],
                       next_point[0] - RACE_LINE[idx][0])
        )
        heading_diff = abs(race_direction - params['heading'])
        if heading_diff > 180:
            heading_diff = 360 - heading_diff

        direction_reward = np.exp(-(heading_diff / 30.0))

        line_bonus = proximity_reward * direction_reward

        margin = 0.5 * track_width - distance_from_center
        if margin < 0.1:
            line_bonus *= 0.25

        current_reward *= (1.0 + line_bonus)

        return current_reward

    # ---- INITIALIZE REWARD DEPENDING ON WHETHER ON TRACK ----
    # reward = on_track_reward(reward)

    # ---- COMBINE REWARD COMPONENTS ----
    # reward_ontrack = on_track_reward(reward)    
    reward_steering = steering_reward(reward)  
    reward_step = step_reward(reward)
    reward_throttle = throttle_reward(reward)           
    reward_raceline = raceline_reward(reward)  
    # reward_proximity = proximity_reward(reward_proximity)

    # specialized straight
    if r_value > 0.8:
        reward = (
            (0.8)*reward_steering + 
            (1.0)*reward_step + 
            (0.5)*reward_throttle + 
            (0.5)*reward_raceline 
            # + (0.5)*reward_proximity
            # if the throttle reward becomes less significant, the car will start drifting again
        )
        
    # specialized curve
    elif r_value < 0.5:
        reward = (
            (0.5)*reward_steering + 
            (0.5)*reward_step + 
            (1.0)*reward_throttle + 
            (1.0)*reward_raceline 
            # + (0.5)*reward_proximity
            # if the throttle reward becomes less significant, the car will start drifting again
        )
    # transition
    else:
        reward = (
            (0.5)*reward_steering + 
            (0.7)*reward_step + 
            (0.8)*reward_throttle + 
            (0.7)*reward_raceline 
            # + (0.5)*reward_proximity
            # if the throttle reward becomes less significant, the car will start drifting again
        )
    return reward