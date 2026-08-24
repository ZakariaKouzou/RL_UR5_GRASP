"""
Task : spawn the UR5e arm and open the interactive 3D viewer.
used to test the installion and paths to the MuJuCo managerie library 
test spawning the robot correctly 

"""

import os
import time
import mujoco
import mujoco.viewer

MENAGERIE_PATH = os.environ.get(
    "MUJOCO_MENAGERIE_PATH",
    os.path.expanduser("~/mujoco_menagerie"),
)
UR5E_XML = os.path.join(MENAGERIE_PATH, "universal_robots_ur5e", "ur5e.xml")

print(f"Loading arm from: {UR5E_XML}")
model = mujoco.MjModel.from_xml_path(UR5E_XML)
data = mujoco.MjData(model)

print("Opening viewer... close the window to end the script.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        # keep it roughly real-time
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Viewer closed.")