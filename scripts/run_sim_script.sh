#export ROS_LOCALHOST_ONLY=1
#unset LD_LIBRARY_PATH
#export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml

# Please update `ISAAC_SIM_PATH` to your local path.
#ISAAC_SIM_PATH=/home/robot/.local/share/ov/pkg/isaac_sim-2022.1.1/

# export ROS_DOMAIN_ID=17
#export ROS_LOCALHOST_ONLY=1
#export ROS_DOMAIN_ID=128


# Please update `ISAAC_SIM_PATH` to your local path.
if [ -z "$ISAAC_SIM_PATH" ]; then
    echo "The environment variable ISAAC_SIM_PATH does not exist!"
else
    $ISAAC_SIM_PATH/python.sh $*
fi

