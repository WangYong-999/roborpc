# RoboRPC

RoboRPC is a Python framework for controlling real and simulated robots through a common interface and ZeroRPC. It includes robot adapters, keyboard/SpaceMouse/Quest/Dynamixel teleoperation, camera integration, cuRobo kinematics and motion planning, and demonstration collection for imitation learning.

## About

Python/ZeroRPC interfaces for real and simulated robot control, teleoperation, and imitation-learning data collection. Includes RealMan, Franka Panda, Flexiv, and UR adapters, plus Isaac Sim and MuJoCo backends; implementation status is documented below.

## Robot support and operation modes

The table describes the checked-in implementation. An adapter file or method name alone does not imply that every operation works. Install the selected SDK and check the configuration and limitations below before using an example.

| Robot / backend | Connection | Implemented operation path | Current limitations |
| --- | --- | --- | --- |
| **RealMan** | Bundled RealMan API in `thirty_party/realman`; dispatched by IDs starting with `realman` | `set_joints()` for a joint target; `set_robot_state()` for joint waypoint streaming; `set_gripper()`; joint feedback | Driver currently selects **ECO65 at `192.168.1.18`**, independently of the adapter IP. Cartesian commands raise `NotImplementedError`; Cartesian feedback is a zero placeholder and gripper feedback needs correction. |
| **Franka Panda** | Polymetis robot/gripper services; dispatched by IDs starting with `panda` | Joint impedance tracking and gripper commands through `set_robot_state()`; joint and end-effector feedback | Dedicated `set_joints()`, `set_ee_pose()`, and `set_gripper()` are stubs. Config uses `franka`, while dispatch looks up `panda`. Gripper connection/state calls need checking against the installed SDK. |
| **Flexiv** | `flexivrdk.Robot`; dispatched by IDs starting with `flexiv` | Non-real-time joint position control; joint impedance via `set_joint_impedance()` and `set_joints()`; SDK state feedback | Default Flexiv config is missing. Cartesian/gripper setters are stubs. Aggregate state reads call unfinished pose/gripper helpers; use SDK state reads directly. |
| **Universal Robots (UR)** | `ur-rtde` control/receive interfaces; Robotiq TCP gripper on port `63352` | `set_robot_state()` sends `servoJ` joint commands; joint/TCP feedback; Robotiq command path | Direct adapter only: absent from `MultiRobots` dispatch. Dedicated motion setters are stubs. Gripper commands expect a scalar rather than the list convention elsewhere. |
| **Dynamixel / GELLO leader** | Serial Dynamixel driver and bundled SDK | Calibrated joint/gripper readings; `set_joints()` and `set_torque_mode()`; leader-to-follower teleoperation through `DynamixelController` | Connects with torque disabled. Cartesian/dedicated gripper setters are stubs. Servo IDs, signs, offsets, and gripper calibration must match the device. |
| **Isaac Sim / Franka** | `SimIsaacRobot` exposes `SimRobotRpcInterface` over ZeroRPC | Joint/gripper actions via `set_robot_state()`; articulation feedback and reset paths | Experimental: camera creation is commented out; the loop expects task observations absent from the generic robot setup. Runtime creates Franka robots; included UR5e assets are not a selectable control backend. |
| **MuJoCo / robosuite Panda** | `SimMujocoRobot` exposes the same RPC interface | Joint/gripper actions via `set_robot_state()`; joint/gripper observations; RGB/depth camera reads and reset | Fixed to **Panda + Lift + JOINT_POSITION**. Requires custom robosuite APIs such as `step(..., set_qpos=...)`. Parsed task/layout/style arguments do not change the constructed environment. |
| **xArm** | Interface skeleton | No robot operations implemented | Methods are stubs, required abstract methods are missing, and there is no dispatcher registration. Not an operational backend. |

Source adapters: [RealMan](roborpc/robots/realman.py), [Panda](roborpc/robots/panda.py), [Flexiv](roborpc/robots/flexiv.py), [UR](roborpc/robots/ur.py), [Dynamixel](roborpc/robots/dynamixel.py), [xArm](roborpc/robots/xarm.py), [Isaac Sim](roborpc/sim_robots/isaac_sim/sim_isaac_robot.py), and [MuJoCo](roborpc/sim_robots/mujoco/sim_mujoco_robot.py).

## Installation

The existing dependency pins target the Python 3.8 era. Use a dedicated environment and install the SDKs needed for your selected backend separately.

```bash
git clone https://github.com/WangYong-999/roborpc.git
cd roborpc
conda create -n roborpc python=3.8
conda activate roborpc
pip install -r requirements.txt
pip install -e .
```

| Feature | Additional requirements |
| --- | --- |
| RealMan | Bundled native libraries in `thirty_party/realman`; matching driver model/IP |
| Panda | Polymetis, PyTorch, and running robot/gripper services |
| Flexiv | Compatible `flexivrdk` SDK and robot connection |
| UR + Robotiq | `ur-rtde` (listed in requirements) and the gripper TCP service |
| Dynamixel / GELLO | Bundled Dynamixel SDK, serial access, and leader calibration |
| SpaceMouse / keyboard | `pyspacemouse`, `pynput`, and a graphical input session; the keyboard adapter currently also opens a SpaceMouse |
| Quest | Oculus reader at `thirty_party.oculus_reader.oculus_reader.reader` (not included in this checkout) and a connected headset |
| `RobotEnv` / Cartesian control | `gym`, CUDA-capable PyTorch, cuRobo, and matching robot descriptions |
| Cameras / trajectory writing | RealSense or ZED SDK for the selected camera; `h5py` for HDF5 collection |
| Simulation | Compatible legacy `omni.isaac` Isaac Sim runtime, or the custom robosuite fork below |

`requirements.txt` is not a complete environment lock. [multi_robots.py](roborpc/robots/multi_robots.py) imports Panda, RealMan, and Flexiv unconditionally, so importing the composed robot/environment stack requires those imported dependencies, including `flexivrdk`, even when selecting another robot.

For the original CUDA 11.8 / PyTorch setup and cuRobo:

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# Run from the directory where you keep third-party source checkouts.
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
```

cuRobo is not pinned here; select a compatible revision for the CUDA/PyTorch stack and APIs used in this project. The MuJoCo backend uses the author's robosuite fork:

```bash
git clone https://github.com/WangYong-999/robosuite.git
cd robosuite
pip install -r requirements.txt
pip install -e .
```

## Configuration and connection modes

[configuration.toml](roborpc/common/config/configuration.toml) is loaded directly by [config_loader.py](roborpc/common/config_loader.py). Edit it before starting a process; there is no command-line config-file override.

For a single RealMan device, replace the corresponding existing sections with:

```toml
[roborpc.robots]
robot_ids = [["realman_1"]]
server_ips_address = ["127.0.0.1"]
sever_rpc_ports = [""]

[roborpc.robots.realman.realman_1]
ip_address = "192.168.1.18"
robot_arm_dof = 6
robot_gripper_dof = 1
start_arm_joints = [0.0, 0.0, -1.5707963267948966, 0.0, 1.5707963267948966, 0.0]
start_gripper_joints = [0.0]
```

This matches the hard-coded ECO65 driver. For another model/IP, also update [DriverRealman](thirty_party/realman/realman_driver.py); changing TOML alone does not change the device it opens. Its constructor releases the gripper and moves to an initial pose. Panda connection likewise calls `go_home()` and opens the gripper.

| Mode | How to connect |
| --- | --- |
| Direct SDK | Instantiate an individual adapter and call `connect_now()`; see the examples below. |
| Local composed access | An empty robot `sever_rpc_ports` entry selects in-process `MultiRobots` through `ComposedMultiRobots`. |
| Remote RPC | Set a nonempty robot port, e.g. `sever_rpc_ports = ["4098"]`, and run `python -m roborpc.robots.multi_robots` on the hardware host. Client `ComposedMultiRobots.connect_now()` invokes the server's hardware connection. |
| Simulation | Start a simulation backend instead of the real-robot server; point robot and camera clients at its RPC endpoint. |

The spelling **`sever_rpc_ports`** is used by robot, controller, and simulation code; cameras use **`server_rpc_ports`**. Separate robot/camera/controller servers on one host need different ports. A simulation server exposes robot and camera interfaces from one endpoint. Controllers also accept an empty port for local use or a nonempty port for RPC (`python -m roborpc.controllers.multi_controllers`).

The composed interfaces accept multiple hosts/IDs, but local factories read only the first nested ID group and composed registries are keyed by server IP. Review grouping before extending a single-robot setup. Other multi-robot execution limitations are described under Environment below.

### Common action format

```python
action = {
    "panda_1": {
        "joint_position": [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0],
        "gripper_position": [0.0],
    }
}
blocking = {"panda_1": {"joint_position": False, "gripper_position": False}}
```

Joint angles are in radians. cuRobo uses Cartesian poses `[x, y, z, roll, pitch, yaw]` in meters/radians; low-level feedback varies: UR returns position plus quaternion, while RealMan's pose is a placeholder. Panda/RealMan gripper targets use `[0.0]` for open and `[1.0]` for closed. Simulation maps targets through configured ranges; UR's command path currently takes a scalar. Gripper feedback is not consistently normalized across adapters.

`set_robot_state()` takes the nested robot/action dictionary on composed/simulation interfaces and only the inner action dictionary on individual adapters. Blocking behavior depends on the driver/method. Composed `set_joints()`, `set_gripper()`, and `set_ee_pose()` require `action_space` and `blocking` dictionaries keyed by robot ID, despite their scalar defaults. Several composed getter methods return empty cached dictionaries or mishandle DOFs; inspect the individual driver or `get_robot_state()` path instead.

## Operating each robot

These examples illustrate implemented call paths and assume the dependencies/configuration above are ready. They have not been validated on physical hardware in this documentation update.

### RealMan: joint target, trajectory, and gripper

```python
from roborpc.robots.realman import RealMan

robot = RealMan("realman_1", "192.168.1.18")
robot.connect_now()
q = robot.get_joint_positions()
robot.set_joints(q)  # Single target; underlying Movej_Cmd waits for completion.
robot.set_robot_state(
    {"joint_position": [q, q]},  # Waypoints: (steps, arm_dof).
    {"joint_position": False},
)
robot.set_gripper([0.0], blocking=True)  # Open; [1.0] closes.
```

The trajectory path streams waypoints with `Movej_CANFD`; a flat vector is not the expected shape. Its gripper state branch skips values near zero, so use `set_gripper([0.0])` to open. Cartesian commands reach an unimplemented driver method. `get_gripper_opening()` derives its value from `state.temperature`, so it is not reliable position feedback.

### Franka Panda: joint impedance tracking

For composed access, rename `[roborpc.robots.franka.panda_1]` to `[roborpc.robots.panda.panda_1]` (likewise for other IDs) to match dispatch. Direct access bypasses that lookup:

```python
from roborpc.robots.panda import Panda

robot = Panda("panda_1", "192.168.1.20")
robot.connect_now()  # go_home(), start_joint_impedance(), and gripper initialization.
q = robot.get_joint_positions()
robot.set_robot_state({"joint_position": q}, {"joint_position": False})
robot.set_robot_state({"gripper_position": [0.0]}, {"gripper_position": False})
print(robot.get_ee_pose())
```

The joint path uses `update_desired_joint_positions(torch.tensor(q))`. Gripper targets become widths using `0.09 * (1 - target)`. Dedicated motion setters do nothing. Reconcile the adapter's `GripperInterface.connect()` and `get_state()` usage with your installed SDK before gripper initialization/feedback.

### Flexiv: joint position and impedance

Add the matching config table; for composed use also select `robot_ids = [["flexiv_1"]]`:

```toml
[roborpc.robots.flexiv.flexiv_1]
ip_address = "YOUR_FLEXIV_ADDRESS"
robot_arm_dof = 7
robot_gripper_dof = 1
```

```python
from roborpc.robots.flexiv import Flexiv

robot = Flexiv("flexiv_1", "YOUR_FLEXIV_ADDRESS")
robot.connect_now()  # Clear faults, enable, wait for operational state.
try:
    q = robot.get_joint_positions()
    robot.set_joints(q, blocking=False)  # NRT_JOINT_POSITION.
    robot.set_joint_impedance([100.0] * 7, [0.7] * 7)
    robot.set_joints(q, blocking=False)  # NRT_JOINT_IMPEDANCE.
    print(robot.robot.states().q, robot.robot.states().dq)
finally:
    robot.disconnect_now()  # Stop.
```

Impedance mode ignores blocking waits. Use explicit stiffness/damping lists: the adapter's default attributes contain trailing commas and become tuples. The position-mode blocking helper also needs correction for list arithmetic, so the example uses `blocking=False`. Cartesian/gripper commands remain unimplemented. Aggregate state reads call pose/gripper helpers that still need SDK integration.

### Universal Robots: RTDE joint servo and Robotiq

Use the adapter directly; `MultiRobots` does not register UR IDs:

```python
from roborpc.robots.ur import UR

robot = UR("ur_1", "192.168.1.10")
robot.connect_now()
q = robot.get_joint_positions()
robot.set_robot_state({"joint_position": q}, {"joint_position": False})
print(robot.get_ee_pose())
# With an operational Robotiq TCP gripper:
robot.set_robot_state({"gripper_position": 0.0}, {"gripper_position": False})
```

The joint command uses `servoJ` with a 2 ms period and `initPeriod()` / `waitPeriod()`; continuous tracking needs repeated targets. The gripper branch scales a scalar target to `0..255`. Cartesian commands and explicit disconnect cleanup are stubs.

### Dynamixel / GELLO: leader teleoperation

Configure `[roborpc.robots.dynamixel.<controller_id>]` with servo IDs, serial port, baud rate, offsets, signs, start joints, and `[servo_id, open_degrees, close_degrees]` gripper calibration. Existing IDs include `dynamixel_controller_panda_1` and `dynamixel_controller_realman_1`.

```python
import numpy as np
from roborpc.common.config_loader import config
from roborpc.robots.dynamixel import Dynamixel

controller_id = "dynamixel_controller_panda_1"
cfg = config["roborpc"]["robots"]["dynamixel"][controller_id]
leader = Dynamixel(
    robot_id=controller_id,
    joint_ids=cfg["joint_ids"],
    joint_offsets=cfg["joint_offsets"],
    joint_signs=cfg["joint_signs"],
    port=cfg["port"],
    gripper_config=cfg["gripper_config"],
    start_joints=np.array(cfg["start_joints"]),
    baudrate=cfg["baudrate"],
)
leader.connect_now()  # Torque disabled for manual movement.
print(leader.get_robot_state())
```

Select the leader ID in `[roborpc.controllers].controller_ids` and follower in `control_robot_ids`. `MultiControllers` calibrates the leader and constructs `DynamixelController`; `forward()` returns follower joint/gripper targets. Direct motor actuation uses `set_torque_mode(True)` and `set_joints()`, but its setter applies offsets without the read path's sign conversion; verify calibration before actuator use.

## Teleoperation

Set `use_controller = true` under `[roborpc.robot_env]`. A local-controller configuration is:

```toml
[roborpc.controllers]
controller_ids = [["spacemouse_controller_1"]]
control_robot_ids = [["panda_1"]]
server_ips_address = ["127.0.0.1"]
sever_rpc_ports = [""]
```

| Controller ID prefix | Operation | Collection signals / prerequisites |
| --- | --- | --- |
| `keyboard_controller` | Arrows move X/Y; `;` / `.` move Z; `e` / `r`, `y` / `h`, `p` / `o` rotate; Space toggles gripper | Scroll Lock = success/start, Pause = failure. Currently also opens a SpaceMouse. |
| `spacemouse_controller` | Six-axis translation/rotation; button 0 sets gripper target `1`, button 1 sets `0` | Scroll Lock = success/start, Pause = failure; needs a kinematics solver. |
| `quest_controller` | Hold selected-hand trigger above `0.5` for relative motion; right-hand B/A set gripper `1`/`0`, left-hand Y/X likewise | Scroll Lock = success/start, Pause = failure. Requires the missing Oculus reader. Open/close labels differ from the Panda/RealMan target convention. |
| `dynamixel_controller` | Move the calibrated leader arm/gripper to generate follower targets | Page Up = success/start, Page Down = failure; leader/follower dimensions must match. |

Keyboard, SpaceMouse, and Quest need compatible kinematics. Pose handling currently differs between components: cuRobo returns a six-value XYZ/Euler vector, while keyboard/SpaceMouse include rotation-matrix operations. Reconcile these paths before Cartesian teleoperation. `MultiControllers.connect_now()` returns inside its loop, so it only connects the first configured controller.

## Simulation

### Isaac Sim

Use the Isaac Sim launcher; this backend imports the legacy `omni.isaac` API:

```bash
export ISAAC_SIM_PATH=/absolute/path/to/isaac-sim
bash scripts/run_sim_script.sh roborpc/sim_robots/isaac_sim/sim_isaac_robot.py
```

Configure `[roborpc.sim_robots.isaac_sim]` and per-robot tables: joint counts, initial joints, stiffness/damping, gripper ranges, and `base_word_pose`. Runtime instantiates the built-in Franka asset per ID. `scene_usd_path` loads the scene; `robot_usd_path` is computed but not used to instantiate the robot. Panda/Robotiq, multi-Panda, and UR5e assets are included as resources. Task examples are under [isaac_sim/tests](roborpc/sim_robots/isaac_sim/tests).

Scene/task observation wiring and camera initialization need completion before a full control/collection loop works; see the support table.

### MuJoCo / robosuite

After installing the custom fork, configure `[roborpc.sim_robots.mujoco]` and run from the RoboRPC checkout:

```bash
python -m roborpc.sim_robots.mujoco.sim_mujoco_robot
```

This starts Panda Lift with joint-position control. Both joint and gripper targets must be present for stepping. The Down arrow requests scene reset. Changing task/robot requires editing `robosuite.make()`; parsed `--task`, `--layout`, and `--style` do not select another environment.

### Simulation RPC client

Set robot `sever_rpc_ports` and camera `server_rpc_ports` to the simulation port (default `4098`) and both addresses to the simulator host. Do not start separate real-robot/camera servers on that endpoint.

```python
import zerorpc

client = zerorpc.Client(heartbeat=20)
client.connect("tcp://127.0.0.1:4098")
try:
    client.connect_now()
    state = client.get_robot_state()
    q = state["panda_1"]["joint_position"]
    client.set_robot_state(
        {"panda_1": {"joint_position": q, "gripper_position": [0.0]}},
        {"panda_1": {"joint_position": False, "gripper_position": False}},
    )
finally:
    client.close()
```

Use `set_robot_state()`; dedicated joint/Cartesian/gripper setters are placeholders. MuJoCo exposes RGB/depth via `read_camera()`; Isaac Sim camera creation currently needs restoring.

## Environment, cameras, and data collection

[RobotEnv](roborpc/robot_env.py) combines robots, cameras, and optional controllers. `get_observation()` returns **`(robot_observations, camera_observations)`**. Pass only robot observations to `controllers.forward()`:

```python
import time
from roborpc.robot_env import RobotEnv

env = RobotEnv()  # Requires configured robots/cameras and use_controller = true.
while True:
    started = time.monotonic()
    robot_obs, camera_obs = env.get_observation()
    action = env.controllers.forward(robot_obs)
    print(action)
    # Enable after validating the selected adapter/controller paths:
    # env.step(action)
    time.sleep(max(0.0, 1.0 / env.env_update_rate - (time.monotonic() - started)))
```

`step()` converts Cartesian targets through cuRobo and calls `set_robot_state()`. It selects kinematics only for RealMan/Panda. The current action-building loop retains only the last robot, and its joint path reduces trajectories to a flat vector, incompatible with RealMan's waypoint-streaming state path. Use the individual RealMan methods until that integration is corrected. Real-robot `reset()` also relies on dedicated setters that are stubs for Panda.

Camera adapters are [RealSense](roborpc/cameras/realsense_camera.py) and [ZED](roborpc/cameras/zed_camera.py). For local dispatch, use IDs starting with `realsense_camera` or `zed_camera`, with matching device configuration. Generic default IDs like `camera_1` refer to simulation cameras and do not match real-camera dispatch. Start a standalone camera service with `python -m roborpc.cameras.multi_cameras` after assigning a separate port.

[DataCollector](roborpc/collector/data_collector.py) writes HDF5 trajectories and separates success/failure episodes using controller signals. Configure `save_data_dir`, `horizon`, cameras, and controller first. Once the selected backend integration is working:

```python
from roborpc.robot_env import RobotEnv
from roborpc.collector.data_collector import DataCollector

env = RobotEnv()
collector = DataCollector(env=env)
collector.collect_trajectory(action_interpolation=False)
```

See [data_collection.md](docs/data_collection.md) for trajectory layout and [data conversion](roborpc/data_convert) for conversion/visualization code. Existing [hardware](docs/hardware_preparation.md) and [software](docs/software_preparation.md) preparation pages are placeholders.

## Kinematics and motion planning

[CuroboSolverKinematic](roborpc/kinematics_solver/curobo_solver_kinematic.py) provides forward/inverse kinematics and collision queries. [CuroboPlanner](roborpc/motion_planning/curobo_planner.py) provides Cartesian-goal planning with `plan()` and joint-goal planning with `plan_js()`. [Robot descriptions](roborpc/robot_description) include single/dual Franka and RealMan models.

Verify model/asset paths and cuRobo compatibility first: the RealMan YAML references `rm_75_6f_description.urdf`, which is absent from this checkout, and the solver's Flexiv branch references a missing description directory. The planner requires `[roborpc.motion_planning.curobo_planner]`, absent from default TOML. Collision-check calls in `RobotEnv.step()` are commented out; not every command path performs collision checking.

## License

[MIT](LICENSE). Bundled third-party SDKs and assets retain their respective licenses.
