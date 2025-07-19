from etils import epath

# feet

FEET_SITES = [
    "ankle_L_sole",
    "ankle_R_sole",
]

LEFT_FEET_GEOMS = [
    "ankle_L_collision",
]

RIGHT_FEET_GEOMS = [
    "ankle_R_collision",
]

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

# base sensors

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
