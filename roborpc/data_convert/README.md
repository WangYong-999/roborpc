# DROID Dataset Conversion

This repo demonstrates how to convert an existing DROID dataset into RLDS format for X-embodiment experiment integration.

## Installation

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow`, 
`tensorflow_datasets`, `matplotlib`, `plotly`, `cv2` and `wandb`.


## Exporting MP4 Data from the DROID Collection Setup

During teleoperation, DROID stores camera data in ZED's `.svo` data format. For conversion to RLDS we first need 
to convert these into `.mp4` files. The easiest is to do this directly on the computer that is running the DROID 
data collection program and controlling the robot, since it already has the necessary [ZED python API](https://www.stereolabs.com/docs/app-development/python/install)
installed.

On the robot machine, run the [`svo_to_mp4.py`](https://github.com/AlexanderKhazatsky/R2D2/blob/main/scripts/convert/svo_to_mp4.py)
script. Afterward, you should see `.mp4` video files in your data directory -- check that the videos look as expected. 
Each video shows both stereo camera views, horizontally concatenated. Next, copy the full data folder to the machine
on which you want to run RLDS conversion below. If you're running RLDS conversion directly on the robot machine, you
do not need to copy the data.


## Converting your Own DROID Dataset to RLDS

You can modify the provided example to convert your own data. Follow the steps below:

1.**Modify Dataset Splits**: The function `_split_paths()` determines the splits of the generated dataset (e.g. training, validation etc.).
If your dataset defines a train vs validation split, please provide the corresponding filenames, e.g. 
by pointing the `crawler` to the corresponding folders (i.e. the data folder you copied from the robot machine above). 
If your dataset does not define splits, put all file names into the `train` split.

2.**Modify Dataset Conversion Code**: We currently assume a fixed dummy language instruction for the full dataset. 
Please modify to use one or multiple appropriate instructions.

That's it! You're all set to run dataset conversion. Run the following:
```
pip3 install -e .
cd droid
tfds build --overwrite
```
The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/tensorflow_datasets/droid`.

**Note 1**: This will overwrite any existing dataset, so make sure to copy previously converted data to a new place or 
add a `--data_dir=<your_new_path>` to the `build` command to write the output to a different folder.

**Note 2**: Conversion uses multi-threading. You can adjust the number of workers and number of episodes held in memory in parallel
via the `N_WORKERS` and `MAX_PATHS_IN_MEMORY` values in `droid.py`.

If you would like to submit your data to the [Open X-Embodiment](https://robotics-transformer-x.github.io/) dataset to help the community, you can add a citation and 
license as described below. Please submit your data via the [enrollment form](https://docs.google.com/forms/d/e/1FAIpQLSeYinS_Y5Bf1ufTnlROULVquD4gw6xY_wUBssfVYkHNaPp4LQ/viewform) and send a short email indicating your submission 
to [`open-x-embodiment@googlegroups.com`](mailto:open-x-embodiment@googlegroups.com).

3.**Provide Dataset Description**: Next, add a bibtex citation for your dataset in `CITATIONS.bib` and add a short description
of your dataset in `README.md` inside the dataset folder. You can also provide a link to the dataset website and please add a
few example trajectory images from the dataset for visualization.

4.**Add Appropriate License**: Please add an appropriate license to the repository. 


## Visualize Converted Dataset
To verify that the data is converted correctly, please run the data visualization script from the base directory:
```
python3 visualize_dataset.py droid
``` 
This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension.
Note, if you are running on a headless server you can modify `WANDB_ENTITY` at the top of `visualize_dataset.py` and 
add your own WandB entity -- then the script will log all visualizations to WandB.

