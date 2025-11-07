import math
import numpy as np

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
    # --- required params (no .get) ---
    is_offtrack   = params['is_offtrack']
    is_crashed    = params['is_crashed']
    is_reversed   = params['is_reversed']
    waypoints     = params['waypoints']
    closest_wps   = params['closest_waypoints']
    heading       = float(params['heading'])
    x             = float(params['x'])
    y             = float(params['y'])
    steps         = int(params['steps'])
    progress      = float(params['progress'])
    steering      = abs(float(params['steering_angle']))
    speed         = float(params['speed'])

    # hard gate
    if is_offtrack or is_crashed or is_reversed:
        return 1e-3

    # straightness
    wp_len = len(waypoints)
    i0, i1 = closest_wps[0] % wp_len, closest_wps[1] % wp_len
    i2 = (i1 + 2) % wp_len

    w0 = np.array(waypoints[i0], dtype=float)
    w2 = np.array(waypoints[i2], dtype=float)
    v  = w2 - w0
    track_dir = math.degrees(math.atan2(v[1], v[0]))
    hd = abs(track_dir - heading)
    if hd > 180.0: hd = 360.0 - hd
    straight_hint = math.cos(math.radians(hd))  # 1 aligned, 0 perp

    # raceline proximity and alignment
    car   = np.array([x, y], dtype=float)
    dists = np.linalg.norm(RACE_LINE - car, axis=1)
    j     = int(np.argmin(dists))
    min_dist = float(dists[j])

    N = len(RACE_LINE)
    jp1 = (j + 1) % N
    nxt = RACE_LINE[jp1]
    race_dir = math.degrees(math.atan2(nxt[1] - RACE_LINE[j][1],
                                       nxt[0] - RACE_LINE[j][0]))
    hd_rl = abs(race_dir - heading)
    if hd_rl > 180.0: hd_rl = 360.0 - hd_rl

    corridor = 0.20
    prox  = math.exp(-max(0.0, (min_dist / corridor)) * 2.5)  # 0..1
    align = math.exp(-max(0.0, (hd_rl / 30.0)))               # 0..1
    raceline_bonus = 1.0 + 0.6 * (prox * align)               # 1..~1.6

    # curvature
    jm1 = (j - 1) % N
    a = RACE_LINE[jm1]; b = RACE_LINE[j]; c = RACE_LINE[jp1]
    v1 = b - a; v2 = c - b
    dot = float(np.dot(v1, v2))
    crz = float(v1[0]*v2[1] - v1[1]*v2[0])   # z of 2D cross
    turn_angle = abs(math.degrees(math.atan2(crz, dot)))  # 0..180

    # bands
    if   turn_angle < 10.0: band = "straight"
    elif turn_angle < 30.0: band = "medium"
    else:                   band = "tight"

    # minimum speeds
    min_v_tight, min_v_medium, min_v_str = 1.4, 1.9, 2.8
    speed_floor_mult = 1.0
    if band == "tight"    and speed < min_v_tight:  speed_floor_mult *= 0.6
    if band == "medium"   and speed < min_v_medium: speed_floor_mult *= 0.7
    if band == "straight" and speed < min_v_str:    speed_floor_mult *= 0.7
    if steering > 25.0:   speed_floor_mult *= 0.9

    # time pressure
    target_pps = 0.45
    pps  = progress / max(steps, 1)
    pace = max(0.6, min(pps / target_pps, 1.4))

    time_tax = math.exp(-0.0025 * steps)

    # finish bonus
    finish = 1.0
    if progress >= 100.0 and steps > 0:
        finish *= min(1000.0 / steps, 2.0)

    # smoothness
    smooth = 1.0
    if straight_hint > 0.9 and steering < 5.0:
        smooth *= 1.05

    # combine &reward
    reward = 1.0
    reward *= raceline_bonus
    reward *= speed_floor_mult
    reward *= pace
    reward *= time_tax
    reward *= smooth
    reward *= finish

    return reward
