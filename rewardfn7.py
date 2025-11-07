import math
import numpy as np

# Edit: remove speed and brake rewards, and add raceline waypoint proximity and alignment reward.

RACE_LINE = np.array([
    [0.6306910855808379, 2.8061193187598317],
    [0.6336712502600561, 2.6907962084913333],
    [0.64671879844676, 2.5756929069468697],
    [0.6697223050606538, 2.4618398778926056],
    [0.7025150580147865, 2.350225687933456],
    [0.7448758872516801, 2.241775135380155],
    [0.7965292269381135, 2.137327702073386],
    [0.8571445889604701, 2.0376165897940752],
    [0.9263357102737471, 1.9432487198980812],
    [1.0036597543535617, 1.8546862524469439],
    [1.0886172129266198, 1.772230506831474],
    [1.1806536995778907, 1.696009722064841],
    [1.27916562451252, 1.6259727993301363],
    [1.3835122734099072, 1.5618915638414308],
    [1.4930357972437118, 1.5033733456028702],
    [1.6070863658096604, 1.4498832214458561],
    [1.7250419173566693, 1.4007717547047782],
    [1.8463044315529822, 1.3553016080057492],
    [1.9702560300742156, 1.3126661163655196],
    [2.09617545495538, 1.271992203465992],
    [2.2231516963031606, 1.2323237433044643],
    [2.355769763848039, 1.190680995904921],
    [2.488141560880269, 1.1483605933912393],
    [2.620103720427086, 1.1049142988426883],
    [2.7515551494884045, 1.060066677008878],
    [2.882456931209304, 1.013714110373173],
    [3.012839289948518, 0.9659425244006475],
    [3.1427840323592005, 0.9169779495926159],
    [3.272395377041519, 0.8671061591514733],
    [3.4017887089897707, 0.8166419369705296],
    [3.525342917396783, 0.7679858244272988],
    [3.64882806345605, 0.720987171881005],
    [3.772250536689716, 0.6765777939272573],
    [3.8956574404119717, 0.63576533247025],
    [4.019126284507557, 0.5993530181742783],
    [4.142731944487423, 0.5680280672308169],
    [4.266522653410114, 0.5424025411418923],
    [4.3905079951724115, 0.5230312892694833],
    [4.514656201929046, 0.5104131135298673],
    [4.6388964110707045, 0.5049812580303754],
    [4.763122708959208, 0.5070893294535697],
    [4.887198566945313, 0.5169978797966641],
    [5.010961461768899, 0.534865307431222],
    [5.134227886556341, 0.5607448881873455],
    [5.256798894774465, 0.5945880421957306],
    [5.378466104203227, 0.6362527456138463],
    [5.499017908693244, 0.6855154688591929],
    [5.618245558333863, 0.742085046996655],
    [5.735948761788441, 0.8056172033729188],
    [5.851940508679295, 0.8757288230719185],
    [5.966050880918649, 0.952011384173886],
    [6.078129704479884, 1.034043172514096],
    [6.188047977606734, 1.1214000482485447],
    [6.295698090655126, 1.2136646292964919],
    [6.400992920613838, 1.31043382985293],
    [6.503863935081087, 1.411324753333731],
    [6.604258473138035, 1.5159789921070206],
    [6.702136383470799, 1.6240654306967688],
    [6.797466194882073, 1.7352816828754838],
    [6.890220974638877, 1.849354314453712],
    [6.980374000991022, 1.9660380119143537],
    [7.067894343462419, 2.085113852939266],
    [7.15274241399219, 2.2063868202399286],
    [7.234865528921659, 2.329682677928094],
    [7.314193510337665, 2.454844303721176],
    [7.390634358086674, 2.581727544764477],
    [7.464070041854949, 2.7101966439988354],
    [7.534352495193162, 2.84011927168856],
    [7.601299937458228, 2.971361196096969],
    [7.664693700697912, 3.10378064039297],
    [7.72427579019867, 3.237222400270733],
    [7.779747452160063, 3.3715118372928403],
    [7.830769051478319, 3.5064489135115737],
    [7.876961568793016, 3.641802488407529],
    [7.917910001805209, 3.777305152780304],
    [7.953168896701898, 3.9126489178386086],
    [7.982270139922662, 4.047482102788967],
    [8.004733011227984, 4.181407762688826],
    [8.020076343394738, 4.313983963934398],
    [8.027832463624136, 4.444726144164974],
    [8.027562422288899, 4.573111687205757],
    [8.018871863543648, 4.698586707129901],
    [8.001426777353679, 4.820574878402188],
    [7.974968309354319, 4.938487985172062],
    [7.939325804964658, 5.05173770861867],
    [7.894427332398623, 5.1597480441847665],
    [7.840307063210617, 5.261967656632944],
    [7.777109078607449, 5.357881452681926],
    [7.705087397848873, 5.4470206857217525],
    [7.624602269446095, 5.528971005317617],
    [7.536113001889473, 5.603378019387619],
    [7.4401678142499055, 5.669950136019715],
    [7.337391337865675, 5.72845867663389],
    [7.228470484463541, 5.778735481212128],
    [7.114139407664787, 5.820668437499423],
    [6.995164227011861, 5.854195539029028],
    [6.872328068022163, 5.879298195009649],
    [6.746416816872122, 5.895994567443778],
    [6.618205817165343, 5.904333692572592],
    [6.48844757334331, 5.9043910566086275],
    [6.357860393423449, 5.896266147481461],
    [6.227117820955166, 5.880082307761797],
    [6.096838683332358, 5.8559889855123615],
    [5.9675776232465605, 5.824166238229436],
    [5.839816075363149, 5.784831109821287],
    [5.713953786027748, 5.738245290891735],
    [5.590301128198999, 5.684723305926886],
    [5.469072612509523, 5.624640360313229],
    [5.350382118821539, 5.558438927015279],
    [5.234240457688271, 5.486633151866064],
    [5.120555941712524, 5.409810146475074],
    [5.0091399384955135, 5.328625034520874],
    [4.899726663769728, 5.243773819450962],
    [4.792013439209298, 5.155933749255134],
    [4.6857631032989016, 5.065605453024112],
    [4.580792695253142, 4.973159174012796],
    [4.480818783513936, 4.882433566317696],
    [4.379597479350928, 4.794203275522037],
    [4.2768435419456345, 4.709068377012267],
    [4.172249406683978, 4.627695662962578],
    [4.065537670885524, 4.550718210850859],
    [3.9564417441231754, 4.478778538106517],
    [3.8447481655301816, 4.412441040406261],
    [3.7303111830280584, 4.352156343601486],
    [3.6130605911709854, 4.298234929044067],
    [3.493004087571274, 4.250828279140158],
    [3.3702246819806065, 4.209917859767956],
    [3.244875114810131, 4.1753097086648605],
    [3.1171707046380055, 4.146633610700183],
    [2.9873824736476857, 4.123344222322313],
    [2.8558513922746442, 4.104669581178079],
    [2.7228205236536547, 4.090057965450789],
    [2.58852157791807, 4.078960262504751],
    [2.453152883048898, 4.070891041745671],
    [2.316756010055542, 4.065796494483948],
    [2.1842507909083038, 4.057872394955421],
    [2.0535654582720935, 4.046523765366982],
    [1.9251617047886112, 4.031055745613201],
    [1.79950824831001, 4.010841840448683],
    [1.6772191072224656, 3.9851132711205137],
    [1.5587352679576263, 3.9534865034022597],
    [1.4446663785956435, 3.9154135169441124],
    [1.3356467086330939, 3.870432395038873],
    [1.23222980790194, 3.818300743675883],
    [1.1351925060030505, 3.7586213016948116],
    [1.0451995262323004, 3.6912827697412762],
    [0.962855346058243, 3.6163720190513566],
    [0.8887074841741225, 3.5341512028508957],
    [0.8232400538850875, 3.445046114429557],
    [0.7668686360501706, 3.3496316889545286],
    [0.7199375587667798, 3.2486142141671337],
    [0.6827192800087478, 3.1428113984019843],
    [0.6554154295486796, 3.033131233951117],
    [0.638159054778358, 2.9205502417869615],
    [0.6306910855808379, 2.8061193187598317]
])

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
    def on_track_reward(current_reward):    # reward if the wheels stay inside the two borders of the track
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
    reward = on_track_reward(reward)

    # ---- COMBINE REWARD COMPONENTS ----
    reward_ontrack = on_track_reward(reward)    
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