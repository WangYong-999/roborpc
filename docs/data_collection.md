# Data Collection



trajectory.h5 project
```text
dataset name: action
- panda_1: Group
  - cartesian_position: (300, 6)
  - gripper_position: (300, 1)
  - joint_position: (300, 7)
dataset name: observation
- camera_1: Group
  - color: (300, 256, 256, 3)
  - depth: (300, 256, 256, 1)
- camera_2: Group
  - color: (300, 256, 256, 3)
  - depth: (300, 256, 256, 1)
- camera_3: Group
  - color: (300, 256, 256, 3)
  - depth: (300, 256, 256, 1)
- panda_1: Group
  - gripper_position: (300, 1)
  - joint_position: (300, 7)
- timestamp: Group
  - control: Group
    - control_start: (300,)
    - policy_start: (300,)
    - sleep_start: (300,)
    - step_end: (300,)
    - step_start: (300,)
```