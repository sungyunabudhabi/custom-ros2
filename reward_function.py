def reward_function(params):
    """
    AWS DeepRacer Reward Function Parameter Unpacking
    """

    # === Core Vehicle and Track State ===
    all_wheels_on_track      = params['all_wheels_on_track']      # Boolean: True if car is fully on track
    x                        = params['x']                        # Float: vehicle’s x-coordinate (m)
    y                        = params['y']                        # Float: vehicle’s y-coordinate (m)
    closest_objects           = params['closest_objects']          # [int, int]: indices of the two closest objects
    closest_waypoints         = params['closest_waypoints']        # [int, int]: indices of the two nearest waypoints
    distance_from_center      = params['distance_from_center']     # Float: distance from track center (m)
    is_crashed                = params['is_crashed']               # Boolean: has the car crashed?
    is_left_of_center         = params['is_left_of_center']        # Boolean: True if left of center line
    is_offtrack               = params['is_offtrack']              # Boolean: True if car is off the track
    is_reversed               = params['is_reversed']              # Boolean: True if driving in reverse direction
    heading                   = params['heading']                  # Float: car heading (yaw) in degrees

    # === Object Detection Parameters ===
    objects_distance           = params['objects_distance']        # [float]: distances of all objects from start line
    objects_heading            = params['objects_heading']         # [float]: headings of each object in degrees (-180,180)
    objects_left_of_center     = params['objects_left_of_center']  # [bool]: list indicating if each object is left of center
    objects_location           = params['objects_location']        # [(float,float)]: (x,y) locations of each object
    objects_speed              = params['objects_speed']           # [float]: speeds of each object (m/s)

    # === Track and Motion Progress ===
    progress                   = params['progress']                # Float: percentage of track completed
    speed                      = params['speed']                   # Float: car speed (m/s)
    steering_angle             = params['steering_angle']          # Float: steering angle in degrees
    steps                      = params['steps']                   # Int: number of steps completed
    track_length               = params['track_length']            # Float: track length (m)
    track_width                = params['track_width']             # Float: width of the track (m)
    waypoints                  = params['waypoints']               # [(float,float)]: list of centerline waypoints

    reward = 1e-3

    if not all_wheels_on_track:
        reward -= 1e-1
    if is_crashed:
        reward -= 1
    if progress > 0.1:
        reward += 0.3
    if speed > 5:
        reward += 0.5
    return float(reward)