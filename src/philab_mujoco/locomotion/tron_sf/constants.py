from etils import epath
from philab_mujoco import ROBOT_PATH

XML = ROBOT_PATH / "SF_TRON1A" / "xml" / "robot.xml"

def task_to_xml(task_name: str) -> epath.Path:
    return XML

# feet

FEET_SITES = [
    "foot_L_site",
    "foot_R_site",
]

LEFT_FEET_GEOMS = [
    "foot_L_collision",
]

RIGHT_FEET_GEOMS = [
    "foot_R_collision",
]

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

# root body

ROOT_BODY = "base_Link"

# base sensors

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
