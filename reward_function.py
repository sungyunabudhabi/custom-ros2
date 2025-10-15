import math
import numpy as np


def reward_function(params):
    """
    Compute a reward for AWS DeepRacer given telemetry parameters.
    """

    # ---- CONSTANTS ----
    MAX_REWARD = 1e2
    MIN_REWARD = 1e-3
    ABS_STEERING_THRESHOLD = 30
    DIRECTION_THRESHOLD = 10

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

    # Initialize reward
    reward = 1.0

    # ---- HELPER FUNCTIONS ----
    def on_track_reward(current_reward):
        if all_wheels_on_track:
            current_reward = MAX_REWARD
        else:
            current_reward = MIN_REWARD
        if is_offtrack:
            current_reward *= 0.1
        if is_crashed:
            current_reward *= 0.01
        if is_reversed:
            current_reward *= 0.1
        return current_reward

    def steering_reward(current_reward):
        next_wp = waypoints[closest_waypoints[1]]
        prev_wp = waypoints[closest_waypoints[0]]
        track_direction = math.degrees(
            math.atan2(next_wp[1] - prev_wp[1], next_wp[0] - prev_wp[0])
        )
        direction_diff = abs(track_direction - heading)
        if abs(steering_angle - direction_diff) > 15.0:
            current_reward *= 0.8  # penalize oversteering
        return current_reward

    def progress_reward(current_reward):
        if steps > 0:
            current_reward += progress / steps
        return current_reward

    def throttle_reward(current_reward):
        if speed > 2.5 - (0.4 * abs(steering_angle)):
            current_reward *= 0.8
        return current_reward

    def speed_reward(current_reward):
        if speed < 1.0:
            current_reward *= 0.5
        elif speed > 3.0:
            current_reward *= 2.0
        return current_reward

    def raceline_reward(current_reward):
        # compute track direction using two waypoints ahead/behind
        wp_len = len(waypoints)

        front_idx = (closest_waypoints[1] + 2) % wp_len
        back_idx = (closest_waypoints[0] + 2) % wp_len

        front_wp = waypoints[front_idx]
        back_wp = waypoints[back_idx]

        track_direction = math.degrees(
            math.atan2(front_wp[1] - back_wp[1], front_wp[0] - back_wp[0])
        )

        # right curve
        if (
            (track_direction - heading) < 0
            and not is_left_of_center
            and 0.1 < distance_from_center < 0.2
        ):
            current_reward *= 1.1

        # left curve
        elif (
            (track_direction - heading) > 0
            and is_left_of_center
            and 0.1 < distance_from_center < 0.2
        ):
            current_reward *= 1.1

        # straight line into a curve
        if (
            0.03 < distance_from_center < (track_width / 2.0)
            and not is_left_of_center
            and track_direction < 0
        ):
            current_reward *= 1.1
        if (
            0.03 < distance_from_center < (track_width / 2.0)
            and is_left_of_center
            and track_direction > 0
        ):
            current_reward *= 1.1

        return current_reward

    def proximity_reward(current_reward):
        # crude check for objects ahead
        if closest_objects[0] >= 10:
            current_reward *= 1.1

        # if there are moving objects, penalize if slower than us
        if len(objects_speed) > 0 and objects_speed[0] < speed:
            current_reward *= 0.7

        return current_reward

    # ---- COMBINE REWARD COMPONENTS ----
    reward = on_track_reward(reward)
    reward = steering_reward(reward)
    reward = progress_reward(reward)
    reward = throttle_reward(reward)
    reward = speed_reward(reward)
    reward = raceline_reward(reward)
    reward = proximity_reward(reward)

    return reward