import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

_MENAGERIE = os.environ.get(
    "MUJOCO_MENAGERIE_PATH",
    os.path.expanduser("~/mujoco_menagerie"),
)
_UR5E_XML = os.path.join(_MENAGERIE, "universal_robots_ur5e", "ur5e.xml")

spec = mujoco.MjSpec.from_file(_UR5E_XML)

# Classic MuJoCo blue checker ground plane (the one used in most of the
# official example scenes).
tex = spec.add_texture()
tex.name = "groundplane"
tex.type = mujoco.mjtTexture.mjTEXTURE_2D
tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
tex.mark = mujoco.mjtMark.mjMARK_EDGE
tex.rgb1 = [0.2, 0.3, 0.4]
tex.rgb2 = [0.1, 0.2, 0.3]
tex.markrgb = [0.8, 0.8, 0.8]
tex.width = 300
tex.height = 300

mat = spec.add_material()
mat.name = "groundplane"
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
mat.texuniform = True
mat.texrepeat = [5, 5]
mat.reflectance = 0.2

ground = spec.worldbody.add_body()
ground.name = "ground"
ggeom = ground.add_geom()
ggeom.name = "floor"
ggeom.type = mujoco.mjtGeom.mjGEOM_PLANE
ggeom.size = [2.0, 2.0, 0.1]
ggeom.material = "groundplane"

# Small red marker showing the currently active waypoint -- visual only,
# no collision, position updated each time the target changes.
marker = spec.worldbody.add_body()
marker.name = "target_marker"
marker.pos = [0.4, 0.0, 0.5]
mgeom = marker.add_geom()
mgeom.name = "target_marker_geom"
mgeom.type = mujoco.mjtGeom.mjGEOM_SPHERE
mgeom.size = [0.025, 0, 0]
mgeom.rgba = [1, 0, 0, 1]
mgeom.contype = 0
mgeom.conaffinity = 0

model = spec.compile()
data = mujoco.MjData(model)

# Elbow-up "standing" start pose
data.qpos[:6] = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
data.ctrl[:6] = data.qpos[:6]
mujoco.mj_forward(model, data)

ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
marker_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
ctrl_range = model.actuator_ctrlrange[:6]

# mink setup
config = mink.Configuration(model)
config.update(data.qpos)
ee_task = mink.FrameTask(
    frame_name="attachment_site",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=0.0,
    lm_damping=1.0,
)

IK_DT = 0.05
N_SUBSTEPS = max(1, int(IK_DT / model.opt.timestep))  # physics steps per IK solve

CIRCLE_CENTER = np.array([0.4, 0.0, 0.45])
CIRCLE_RADIUS = 0.15
CIRCLE_ANGULAR_SPEED = 0.6  # rad/s

theta = 0.0
target = CIRCLE_CENTER + CIRCLE_RADIUS * np.array([np.cos(theta), 0.0, np.sin(theta)])
model.body_pos[marker_body_id] = target

print("Opening viewer -- the end-effector will trace a circle continuously.")

last_print = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        current_rotation = config.get_transform_frame_to_world(
            "attachment_site", "site"
        ).rotation()
        target_se3 = mink.SE3.from_rotation_and_translation(current_rotation, target)
        ee_task.set_target(target_se3)

        vel = mink.solve_ik(config, [ee_task], IK_DT, "daqp", 1e-3)
        config.integrate_inplace(vel, IK_DT)

        data.ctrl[:6] = np.clip(config.q[:6], ctrl_range[:, 0], ctrl_range[:, 1])

        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(model, data)
        config.update(data.qpos)
        viewer.sync()

        ee_pos = data.site_xpos[ee_site_id]
        dist = np.linalg.norm(ee_pos - target)

        # Live terminal feedback, printed a few times a second
        now = time.time()
        if now - last_print > 0.3:
            print(f"  ee = [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]   "
                  f"target = [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]   "
                  f"dist = {dist:.4f}")
            last_print = now

        # Advance the target along the circle
        theta += CIRCLE_ANGULAR_SPEED * IK_DT
        target = CIRCLE_CENTER + CIRCLE_RADIUS * np.array([np.cos(theta), 0.0, np.sin(theta)])
        model.body_pos[marker_body_id] = target

        elapsed = time.time() - step_start
        time.sleep(max(0.0, IK_DT - elapsed))

print("Viewer closed.")