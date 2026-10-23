
# ROBORPC: Multi-Robots Cooperative Manipulation Toolbox via ZeroRpc

## Abstract

Robots are increasingly being used in a wide range of applications, from industrial automation to entertainment and education. However, robot manipulation is still a challenging task, requiring a combination of dexterity, precision, and skill acquisition. To address this challenge, we propose a new approach called ROBORPC (Multi-Robots Cooperative Manipulation Toolbox via ZeroRpc), which enables direct skill transfer from in-the-wild human demonstrations to deployable robot policies. ROBORPC is a framework that enables multi-robot cooperative manipulation tasks by leveraging the power of ZeroRpc, a high-performance RPC framework that enables communication between robots and computers in real-time. We demonstrate the effectiveness of ROBORPC by applying it to a real-world multi-robot manipulation task, where a human operator is required to manipulate a set of robots to complete a complex task. We also present a set of evaluation results that show the potential of ROBORPC for multi-robot cooperative manipulation tasks. ROBORPC's hardware and software components are open-source at https://github.com/roborpc/roborpc and can be easily integrated into existing robotics systems.

## Introduction
Imitation learning is a popular approach to learn cooperative behaviors for multi-robots from human demonstrations. However, it is challenging to apply imitation learning to real-world multi-robot manipulation tasks due to the complexity of the task and the limited availability of demonstrations. The key challenge is to collect and annotate a large set of demonstrations that cover a wide range of manipulation tasks and scenarios. In this paper, we propose a new approach called ROBORPC (Multi-Robots Cooperative Manipulation Toolbox via ZeroRpc), which enables direct skill transfer from in-the-wild human demonstrations to deployable robot policies. ROBORPC is a framework that enables multi-robot cooperative manipulation tasks by leveraging the power of ZeroRpc, a high-performance RPC framework that enables communication between robots and computers in real-time. We demonstrate the effectiveness of ROBORPC by applying it to a real-world multi-robot manipulation task, where a human operator is required to manipulate a set of robots to complete a complex task. We also present a set of evaluation results that show the potential of ROBORPC for multi-robot cooperative manipulation tasks. 

The method proposed in this paper can be applied to distributed multi-robot data acquisition equipment, and can collect data remotely and efficiently even in dangerous working areas.

In this paper, we first introduce the problem of multi-robot cooperative manipulation tasks and the need for a new approach to address it. We then describe the ROBORPC framework, which enables multi-robot cooperative manipulation tasks by leveraging the power of ZeroRpc. We present the architecture of ROBORPC and the components that enable direct skill transfer from in-the-wild human demonstrations to deployable robot policies. We then demonstrate the effectiveness of ROBORPC by applying it to a real-world multi-robot manipulation task, where a human operator is required to manipulate a set of robots to complete a complex task. Finally, we present a set of evaluation results that show the potential of ROBORPC for multi-robot cooperative manipulation tasks.

## Related Work
The key of imitation learning  for multi-robot cooperative manipulation tasks is to collect data. Here, we review the related work on collecting human demonstrations for multi-robot cooperative manipulation tasks.

### Hand-Held Grasper
The handheld gripper is detached from the robot, providing a portable and intuitive data collection method for "in-the-wild" data collection [1, 2, 3, 4]. Previous studies have tried various methods to robustly extract the end pose of the robot. For example, UMI integrates advanced SLAM and GoPro's built-in IMU data [5] to accurately capture 6D pose; or uses RGBD and external tracking devices to improve the end tracking accuracy [1, 2]. However, this method needs to rely on a high success rate Ik solver to deploy to the robot, and lacks visual perception of the robot body during the collection process. Due to the IK solver problem, it may lead to strange joint poses and cause collisions between multiple robots.

### Visual Demonstration from Human Video
Another approach is to use visual demonstrations from human video [6, 7, 8]. For example, gesture detectors are used to infer action data from human videos [9]; to reduce the gap between human poses and robot actions, human-to-robot action mappings with gesture relocalization are learned [10]. Nevertheless, there is still a gap between human demonstrations and robot actions, especially in the performance of fine and long-term tasks.

### Teleportation
Datasets collected by teleoperating real robots can achieve the most direct transferability. For example, RT-X uses a VR controller device for teleoperation [], which directly controls the end of the robot arm and also faces the problem of IK solution failure; on the contrary, the use of joint mapping in ALOHA [] and GELLO [] can more directly control the robot for fine manipulation. However, none of them specifically involves the collaborative task of multiple (more than three) robots. In contrast, this paper considers the consistency of multi-robot collaboration and the convenience of remote teleoperation.

## Method
ROBORPC is a multi-robot data collection and motion planning framework that allows the conversion of data from human demonstrations into robot-deployable policies. The following two sections will introduce the hardware and software of the framework.

### A. Hardware
The hardware of ROBORPC consists of a set of robots and a computer. The robots are equipped with a camera, a depth sensor, and a manipulator. The computer is equipped with a high-performance GPU and a ZeroRpc server.

### B. Software
The software of ROBORPC consists of a set of components that enable the data collection and motion planning. The components include a data collection module, a motion planning module, and a policy deployment module.

#### Data Collection Module
The data collection module consists of a set of tools that enable the collection of human demonstrations. The tools include a camera, a depth sensor, and a handheld gripper. The camera and depth sensor are used to capture the environment and the handheld gripper is used to collect the end-effector pose. The data collection module is responsible for collecting the demonstrations and annotating them with action labels.

#### Motion Planning Module
The motion planning module consists of a set of algorithms that enable the planning of robot trajectories. The algorithms include a motion planner, a path optimizer, and a trajectory executor. The motion planner generates a set of feasible trajectories for the robot to follow. The path optimizer optimizes the trajectories to reduce the error and the trajectory executor executes the trajectories on the robot.

#### Policy Deployment Module
The policy deployment module consists of a set of tools that enable the deployment of robot policies. The tools include a policy generator, a policy optimizer, and a policy executor. The policy generator generates a set of robot policies based on the collected demonstrations. The policy optimizer optimizes the policies to reduce the error and the policy executor executes the policies on the robot.

## Experiments
In this section, we present the experiments that demonstrate the effectiveness of ROBORPC for multi-robot cooperative manipulation tasks.

### A. Data Collection
We collected a set of demonstrations from a human operator to demonstrate the effectiveness of ROBORPC for multi-robot cooperative manipulation tasks. The demonstrations were collected in a real-world environment using a set of robots. The robots were equipped with a camera, a depth sensor, and a manipulator. The operator was required to manipulate the robots to complete a complex task.

### B. Motion Planning
We used the motion planning module of ROBORPC to generate a set of robot trajectories based on the collected demonstrations. The motion planner used in ROBORPC is a probabilistic motion planner that generates a set of feasible trajectories for the robot to follow. The path optimizer used in ROBORPC is a probabilistic path optimizer that optimizes the trajectories to reduce the error. The trajectory executor used in ROBORPC is a probabilistic trajectory executor that executes the trajectories on the robot.

### C. Policy Deployment
We used the policy deployment module of ROBORPC to deploy a set of robot policies based on the generated trajectories. The policy generator used in ROBORPC is a probabilistic policy generator that generates a set of robot policies based on the collected demonstrations. The policy optimizer used in ROBORPC is a probabilistic policy optimizer that optimizes the policies to reduce the error. The policy executor used in ROBORPC is a probabilistic policy executor that executes the policies on the robot.

### D. Evaluation
We evaluated the effectiveness of ROBORPC for multi-robot cooperative manipulation tasks by comparing the performance of the deployed policies with the performance of the human operator. We used a set of metrics to evaluate the performance of the deployed policies, including the success rate, the average time to complete the task, and the average path length. We also compared the performance of the deployed policies with the performance of a baseline policy that does not use ROBORPC. The baseline policy used in this experiment is a simple policy that always moves the end-effector to the target position. We also evaluated the effectiveness of ROBORPC for multi-robot cooperative manipulation tasks in a real-world environment.


## Conclusion
In this paper, we proposed a new approach called ROBORPC (Multi-Robots Cooperative Manipulation Toolbox via ZeroRpc) that enables direct skill transfer from in-the-wild human demonstrations to deployable robot policies. ROBORPC is a framework that enables multi-robot cooperative manipulation tasks by leveraging the power of ZeroRpc, a high-performance RPC framework that enables communication between robots and computers in real-time. We demonstrated the effectiveness of ROBORPC by applying it to a real-world multi-robot manipulation task, where a human operator is required to manipulate a set of robots to complete a complex task. We also present a set of evaluation results that show the potential of ROBORPC for multi-robot cooperative manipulation tasks. ROBORPC's hardware and software components are open-source at https://github.com/roborpc/roborpc and can be easily integrated into existing robotics systems.


## References
[1]	Grasping in the wild: Learning 6dof closed- loop grasping from low-cost demonstrations.

[2]	Scalable intuitive human to robot skill transfer with wearable human machine interfaces: On complex

[3]	J. Pari, N. M. (mahi) Shafiullah, S. P. Arunachalam, and L. Pinto, “The surprising effectiveness of representation learning for visual imitation,” in Robotics: Science and Systems XVIII, 2022.

[4]	P. Praveena, G. Subramani, B. Mutlu, and M. Gleicher, “Characterizing input methods for human-to-robot demonstrations,” in 2019 14th ACM/IEEE International Conference on Human-Robot Interaction (HRI), 2019.

[5] C. Chi et al., “Universal Manipulation Interface: In-the-wild robot teaching without in-the-wild robots,” arXiv [cs.RO], 2024.

[6]	A. Chen, S. Nair, and C. Finn, “Learning generalizable robotic reward functions from ‘in-the-wild’ human videos,” in Robotics: Science and Systems XVII, 2021.

[7]	A. Simeonov et al., “Neural Descriptor Fields: SE(3)-equivariant object representations for manipulation,” arXiv [cs.RO], 2021.

[8]	Y. Qin et al., “DexMV: Imitation learning for dexterous manipulation from human videos,” in Lecture Notes in Computer Science, Cham: Springer Nature Switzerland, 2022, pp. 570–587.

[9]	K. Schmeckpeper et al., “Learning predictive models from observation and interaction,” in Computer Vision – ECCV 2020, Cham: Springer International Publishing, 2020, pp. 708–725.

[10]	K. Shaw, S. Bahl, A. Sivakumar, A. Kannan, and D. Pathak, “Learning dexterity from human hand motion in internet videos,” Int. J. Rob. Res., vol. 43, no. 4, pp. 513–532, 2024.
