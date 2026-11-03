---
description: This tutorial shows how to build a complete visual grasping demo for the reBot Arm B601 using the YOLO/OBB grasping pipeline.
title: reBot Arm B601 Visual Grasping Demo
keywords:
  - reBot Arm
  - B601
  - Grasping
  - Gemini 2
  - YOLO
  - Hand-Eye Calibration
  - Robot
slug: /rebot_arm_b601_dm_grasping_demo
sku: 100065783, 100095532, 100063143, 100045679, 100040187
last_update:
  date: 2026-06-30
  author: YinHaizhou
translation:
  skip: [zh-CN]
createdAt: '2026-04-22'
updatedAt: '2026-06-30'
url: https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/
---

# reBot Arm B601 Visual Grasping Demo

<p align="center">
  <img src="https://raw.githubusercontent.com/Seeed-Projects/reBot-DevArm/main/media/v1.0.png" alt="reBot Arm B601" />
</p>

<p align="center">
    <a href="./LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" />
    </a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version" />
    <img src="https://img.shields.io/badge/Platform-Ubuntu%2022.04+-orange.svg" alt="Platform" />
    <img src="https://img.shields.io/badge/Camera-RGB--D-green.svg" alt="Camera" />
    <img src="https://img.shields.io/badge/Detection-YOLO-yellow.svg" alt="YOLO" />
</p>

<p align="center">
  <strong>Depth Perception · Object Detection · Hand-Eye Calibration · Autonomous Grasping · Fully Open Source</strong>
</p>

<p align="center">
  <strong>
    <a href="./README_zh.md">简体中文</a> &nbsp;|&nbsp;
    <a href="./README.md">English</a>
  </strong>
</p>

---

## Project Introduction

**reBot Arm B601 Visual Grasping Demo** is a visual grasping algorithm demonstration project based on the [reBot Arm B601](https://github.com/vectorBH6/reBotArm_control_py) robotic arm control library and RGB-D depth camera. The system supports both DM and RS configurations for the B601 arm. It uses the YOLO model for real-time desktop object detection, estimates grasp poses via OBB minimum-area rectangles, performs hand-eye calibration to transform grasp points from the camera frame to the robot base frame, and drives the robotic arm to complete autonomous grasping.

### Core Features

- 📷 **Depth Perception** — Supports RGB-D depth cameras such as Orbbec Gemini 2 and Intel RealSense D435i / D405
- 🔍 **Object Detection** — YOLO-based recognition with open-vocabulary custom class support
- 📐 **Pose Estimation** — OBB minimum-area rectangle short axis for gripper orientation, depth quantile for grasp height estimation
- 🔄 **Coordinate Transformation** — TSAI hand-eye calibration (Eye-in-Hand), transforming camera frame grasp points to robot base frame
- 🦾 **Motion Execution** — reBotArm_control_py IK + trajectory controller with built-in gripper force control state machine

---

## Hardware Configuration

| Component | Model / Requirements |
|------|------------|
| Robotic Arm | reBot Arm B601 (DM / RS configurations) |
| Depth Camera | Orbbec Gemini 2, Intel RealSense D435i / D405 |
| Communication Interface | USB2CAN serial bridge (arm); USB 3.0 (camera) |
| Host | Ubuntu 22.04+, Python 3.10, x86_64 or NVIDIA Jetson Thor (JetPack 7, aarch64) |

**Wiring Instructions**

1. Connect the depth camera to the host via USB 3.0
2. Connect the USB2CAN adapter to the arm CAN bus and insert it into the host USB port
3. Configure device permissions:

```bash
sudo chmod a+rw /dev/bus/usb/*/*   # Depth camera USB permissions
sudo chmod 666 /dev/ttyUSB0        # USB2CAN (adjust port number as needed)
```

---

## Environment Installation

### Step 1. Clone the Repository

Prefer the official Seeed-Projects repository:

```bash
git clone https://github.com/Seeed-Projects/reBot-DevArm-Grasp.git rebot_grasp
cd rebot_grasp
```

You can also use the current development repository:

```bash
git clone https://github.com/EclipseaHime017/reBot-DevArm-Grasp.git rebot_grasp
cd rebot_grasp
```

### Step 2. Create and Configure the conda Environment

```bash
conda env create -f environment.yml
conda activate rebotarm
```
:::tip
If you want to use a different environment name, replace `rebotarm` in the command with your custom name.

:::

**NVIDIA Jetson Thor (JetPack 7, CUDA 13)**

The same `environment.yml` works on Jetson Thor. Two things differ from an x86 PC:

- Install the CUDA 13 build of PyTorch (aarch64 wheels are on the PyTorch index), then check that the GPU is visible:

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```

- Run detection on the GPU: set `yolo.device: "0"` in `config/default.yaml`. YOLOE-26L takes about 30 ms per frame on the Thor GPU, against several seconds on its CPU.

`environment.yml` pins conda-forge's OpenBLAS (`libblas=*=*openblas`): the NVPL BLAS build conda-forge picks on aarch64 makes `import numpy` fail. The pin changes nothing on x86, where OpenBLAS is already the default.


### Step 3. Install the Robotic Arm Control Library

```bash
git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
cd sdk/reBotArm_control_py
pip install -e .
cd ../..
```

If `pip install -e .` reports `Multiple top-level packages discovered in a flat-layout`, add explicit package discovery configuration to `reBotArm_control_py`'s `pyproject.toml`, then re-run `pip install -e .`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["reBotArm_control_py*"]
```

The visual grasping program reads the SDK configuration and automatically selects the corresponding arm control mode and gripper parameters.

### Step 4. Install the Depth Camera SDK

This project supports RGB-D depth cameras such as Orbbec Gemini 2 and RealSense D435i / D405. Please install the corresponding SDK for your actual camera; if the camera driver can already be imported normally in the current environment, you may skip this step.

**Orbbec Gemini 2**

Orbbec Gemini 2 depends on **pyorbbecsdk** (Python version of Orbbec SDK v2). It is recommended to install the prebuilt Python package directly:

**Option 1: Install via pip (recommended)**

```bash
pip install pyorbbecsdk2
```

**Option 2: Get from GitHub**

```bash
# Install build dependencies
sudo apt-get install -y cmake build-essential libusb-1.0-0-dev

cd sdk
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
pip install -e .
```

For mainland China users, you can use:
```bash
git clone https://gitee.com/orbbecdeveloper/pyorbbecsdk.git
```

When installing from source, build the native extension with CMake first to ensure `install/lib` contains `pyorbbecsdk*.so` and Orbbec shared libraries, then run `pip install -e .`.

Note: If all installation methods above fail, please refer to the official Orbbec documentation below for installation.

:::tip
For first-time use, it is recommended to install udev rules:

```bash
sudo bash scripts/install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**Verify Installation**

```bash
python -c "import pyorbbecsdk; print('pyorbbecsdk OK')"
```
:::

**OrbbecViewer (optional, for camera verification)**

After downloading the prebuilt package and running `OrbbecViewer`, you can verify the camera connection and depth stream are working properly before running the Demo.

- GitHub: https://github.com/orbbec/OrbbecSDK_v2/releases
- Gitee: https://gitee.com/orbbecdeveloper/OrbbecSDK_v2/releases

**RealSense D435i / D405**

RealSense cameras depend on `pyrealsense2`. You can usually install it directly via pip:

```bash
pip install pyrealsense2
python -c "import pyrealsense2; print('pyrealsense2 OK')"
```

If the system needs the complete RealSense toolkit or udev rules, please refer to the RealSense SDK official documentation to install `librealsense2`.


**SDK Resource Summary**

| Resource | Link |
|------|------|
| Gemini 2 Product Page | https://www.orbbec.com.cn/index/Product/info.html?cate=38&id=51 |
| Development Resources | https://www.orbbec.com.cn/index/Download2025/info.html?cate=121&id=1 |
| Orbbec SDK v2 | https://github.com/orbbec/OrbbecSDK_v2 |
| SDK v2 API Documentation | https://orbbec.github.io/docs/OrbbecSDKv2_API_User_Guide/ |
| pyorbbecsdk | https://github.com/orbbec/pyorbbecsdk |
| pyorbbecsdk Documentation | https://orbbec.github.io/pyorbbecsdk/index.html |
| ROS2 Wrapper | https://github.com/orbbec/OrbbecSDK_ROS2/tree/v2-main |
| Intel RealSense SDK | https://github.com/realsenseai/librealsense |

### Step 5. Configure GraspNet (optional)

To achieve more accurate grasp pose estimation for objects, this project adapts [graspnet-baseline](https://github.com/graspnet/graspnet-baseline) to improve robotic arm grasping performance.

The GraspNet `pointnet2` / `knn` extensions require a CUDA compiler. Before starting, confirm that `nvcc` is available in the current environment and check that the CUDA version reported by `nvcc` matches the CUDA version used to build PyTorch:

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

When configuring this environment for the first time, the `torchaudio` entry in `environment.yml` usually installs PyTorch along with it. If it is missing (or only the CPU build was installed), follow the steps below to explicitly install and verify a CUDA-enabled PyTorch. The GraspNet `pointnet2` / `knn` extensions depend on PyTorch and a CUDA compiler with a matching version — finish all steps before building.

**Step 1: Confirm your GPU**

```bash
nvidia-smi
```

Continue only if an NVIDIA GPU is listed. Without an NVIDIA GPU you can only use the CPU build of PyTorch, and the GraspNet CUDA operators cannot be built or run.

**Step 2: Check / install PyTorch**

First check whether PyTorch already exists in the current environment:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

- If it is installed and `torch.cuda.is_available()` prints `True`, skip the installation and go to Step 3.
- If it raises `ModuleNotFoundError` (not installed), or `torch.version.cuda` is `None` (CPU build installed), install the CUDA build as shown below. Take CUDA 13.0 as an example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Pick the CUDA version in `--index-url` based on your GPU:

| GPU | Recommended PyTorch CUDA build |
| --- | --- |
| RTX 50 series (Blackwell, e.g. RTX 5070 / 5090) | `cu128` or newer (e.g. `cu130`) |
| RTX 40 / 30 series (Ada / Ampere) | `cu121` or `cu118` |
| No NVIDIA GPU | `cpu` (CPU only, GraspNet will not run) |

**Step 3: Install the CUDA compiler matching your PyTorch**

`pointnet2` / `knn` are local CUDA extensions: the CUDA version reported by `nvcc --version` must match `torch.version.cuda` when building, otherwise the build fails with `The detected CUDA version (...) mismatches the version that was used to compile PyTorch (...)`. Check both versions first:

```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

Take `torch.version.cuda` showing `13.0` as an example, and install the matching compiler inside the conda environment:

```bash
conda install -c nvidia cuda-nvcc=13.0 cuda-cudart-dev=13.0

which nvcc       # should point to $CONDA_PREFIX/bin/nvcc, not /usr/bin/nvcc
nvcc --version   # should match torch.version.cuda
```

`cuda-nvcc` provides the compiler and `cuda-cudart-dev` provides the CUDA headers under `targets/x86_64-linux/include`; both are required. Once the conda environment is activated, `$CONDA_PREFIX/bin` precedes the system paths, so the conda `nvcc` automatically shadows the old system one (for example CUDA 11.5 from the Ubuntu repositories) — there is no need to uninstall the system CUDA.

> **Warning**
> Do not downgrade PyTorch to match an old system `nvcc`: old CUDA releases may not support your GPU architecture (CUDA 11.5 supports at most sm_86; RTX 40 series requires CUDA 11.8+ and RTX 50 series requires CUDA 12.8+), and old PyTorch builds may be incompatible with other project dependencies such as recent ultralytics releases.

**Step 4: Verify**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

When both CUDA versions match and `torch.cuda.is_available()` prints `True`, the PyTorch setup is complete and you can continue building the CUDA operators.

```bash
cd sdk
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline

# Install PyTorch for your CUDA version first, then install GraspNet runtime dependencies
pip install open3d tensorboard Pillow tqdm

# Configure CUDA build paths before building the local operators.
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"
export CPATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/include:$CPATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Build CUDA operators
cd pointnet2
pip install . --no-build-isolation
cd ../knn
pip install . --no-build-isolation
cd ..

# Install GraspNet API
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
sed -i "s/'sklearn'/'scikit-learn'/" setup.py
pip install .
cd ../../..
```

:::tip
Note: If you directly follow the official graspnet-baseline repository documentation and use `python setup.py install`, CUDA / PyTorch related errors may occur. It is recommended to use `pip install . --no-build-isolation` so the extension is built against the PyTorch and CUDA configuration already installed in the current conda environment.
:::

:::tip
If building fails with `fatal error: cusparse.h: No such file or directory`, run `find $CONDA_PREFIX -name cusparse.h` and add the directory containing `cusparse.h` to `CPATH` / `CPLUS_INCLUDE_PATH`. If the CUDA headers come from conda `cuda-toolkit`, the path is usually `$CONDA_PREFIX/targets/x86_64-linux/include`, not the pip `nvidia/cu13/include` path shown above.
:::

:::tip
Additionally, older GraspNet API dependencies may still use the deprecated `sklearn` package name. The `sed` command above replaces it with `scikit-learn` to avoid package name issues during installation. Unless you also upgrade the GraspNet API dependency stack, keep its `numpy==1.23.4` constraint because `transforms3d==0.3.1` still uses NumPy aliases such as `np.float`.
:::

**Configure Pretrained Model**

Download the official GraspNet pretrained weights from the graspnet-baseline official repository [Google](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view), [Baidu](https://pan.baidu.com/s/1Eme60l39tTZrilF0I86R5A), and place the downloaded `checkpoint-rs.tar` at:

```bash
sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar
```

Then verify in `config/default.yaml`:

```yaml
graspnet:
  checkpoint: "checkpoint-rs.tar"
```

The `checkpoint` field supports three forms: a filename only is resolved under `sdk/graspnet-baseline/checkpoints/`; a relative path is resolved from the project root; an absolute path is used directly.

---

## Directory Structure

```
rebot_grasp/
├── config/
│   ├── default.yaml              # Main configuration file
│   └── calibration/
│       └── <camera_type>/
│           ├── intrinsics.npz    # Camera intrinsics
│           └── hand_eye.npz      # Hand-eye calibration results
├── drivers/
│   ├── camera/
│   │   ├── base.py               # Camera abstract base class
│   │   ├── orbbec_gemini2.py     # Gemini 2 driver
│   │   └── realsense.py          # RealSense driver (alternative)
│   └── robot/
│       └── grasp_driver.py       # Lightweight grasping helper based on arm SDK
├── calibration/
│   ├── aruco_pose.py             # ArUco pose estimation
│   └── hand_eye.py               # Hand-eye calibration solver
├── utils/
│   ├── ordinary_grasp.py         # OBB grasp pose estimation and visualization
│   ├── pick_place.py             # Pick-and-place pipeline (used by set.py)
│   ├── reach_map.py              # Reach zones: where the arm can pick / release
│   ├── grasp_check.py            # Wrist-camera check that the gripper holds something
│   └── transforms.py             # Coordinate transformation utilities
├── scripts/
│   ├── main.py                   # Main grasping program
│   ├── set.py                    # Grasp and place program
│   ├── ordinary_grasp_pipeline.py
│   ├── object_detection.py
│   └── collect_handeye_eih.py
├── sdk/
│   ├── pyorbbecsdk/              # Orbbec SDK Python wrapper
│   └── reBotArm_control_py/      # reBot Arm SDK
└── environment.yml               # Recommended conda environment file
```


## Running and Debugging

### 0. Confirm Arm Version and SDK Configuration

Before running scripts that connect to the robotic arm, confirm that the arm version, power supply, and SDK configuration are consistent:

- Please complete the basic arm preparation first: [B601-DM Quick Start](https://wiki.seeedstudio.com/cn/rebot_b601_dm_getting_started/) or [B601-RS Quick Start](https://wiki.seeedstudio.com/cn/rebot_b601_rs_getting_started/).
- In `sdk/reBotArm_control_py/config/rebotarm.yaml`, select the corresponding hardware configuration:

```yaml
hardware_yaml: rebotarm_dm.yaml
```

Or:

```yaml
hardware_yaml: rebotarm_rs.yaml
```

- B601-DM uses 24V DC power, B601-RS uses 48V DC power. Please confirm the power adapter and wiring match the arm version.
- When using B601-DM, confirm the serial bridge device path in the SDK configuration matches the actual device.
- When using B601-RS, start the CAN interface before running calibration or grasping scripts:

```bash
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
ip -details link show can0
```

### 1. Hand-Eye Calibration (Required Before Grasping)

```bash
python scripts/collect_handeye_eih.py
```

In automatic mode, the arm automatically traverses 50 preset poses, and automatically samples when ArUco is detected stably. When finishing normally or being interrupted midway, the script tries to compute and save the calibration result; at least 5 samples are required, with 15 or more recommended for more stable results.

If you want to manually move the arm for collection, use:

```bash
python scripts/collect_handeye_eih.py --manual
```

In manual mode, the arm enters gravity-compensation mode. Push the end effector to a suitable viewing angle and press `Enter` to capture, press `c` or `q` to finish and compute.

:::tip
If you find that the robotic arm's grasping accuracy cannot meet your requirements after calibration, you can set the `X` (front-back), `Y` (left-right), `Z` (up-down) parameters in `config/default.yaml` under `calibration.hand_eye_compensation_m` to provide positional compensation.
:::

### 2. `scripts/main.py` — Main Grasping Program

Complete visual grasping pipeline:

1. Initialize RGB-D camera, confirm image stream is available
2. Enable arm and gripper, move to ready position
3. Real-time camera preview + YOLO object detection and instance segmentation
4. OBB short axis estimates gripper orientation, depth quantile estimates grasp height
5. Press `G` to freeze frame, compute arm target pose via hand-eye transformation
6. Arm moves to pre-grasp point → descends → gripper closes → lifts → returns to ready position

```bash
python scripts/main.py
python scripts/main.py --dry-run          # estimate only, do not move the arm
python scripts/main.py --skip-zero-check  # skip the startup zero check (see below)
```

**Startup zero check.** Before enabling the arm, every motor's position is read (read-only, nothing is enabled) and must be within `robot.zero_check_tol_rad` of zero: the arm folded at its rest pose and the gripper closed, which is where the motors were zeroed. A motor far from zero, often off by a whole turn after a crash or after the arm was moved by hand, would make every commanded pose wrong, so the program prints the readings and stops. Fold the arm to rest and close the gripper, or re-zero the motor that lost its zero. Only RobStride motors (B601-RS) can be read this way; DM motors are listed as not checked.

**Size-aware gripper.** The detection measures the object's width across the jaws. With `robot.gripper.size_aware.enabled`, the gripper opens only `open_margin_m` wider than the object, closes no further than `squeeze_m` below its width (so a soft or mis-measured object is not crushed), and objects too wide for the jaws are skipped. The log shows the plan for each grasp, e.g. `object 4.2 cm: open to 7.2 cm, never close below 3.2 cm`, and why the gripper stopped (`contact`, `stopped at width limit`, `closed fully, no object`).

### 3. `scripts/set.py` — Grasp and Place Program

Function: pick up one object and drop it into or onto another (by default the banana into the box). Both are text labels, so with an open-vocabulary model (YOLOE / YOLO-World) any pair can be chosen while the program runs.

```bash
python scripts/set.py
python scripts/set.py --pick "orange ball" --place "cup" --release-clearance 0.05
python scripts/set.py --dry-run          # detect and plan, do not move the arm
python scripts/set.py --skip-zero-check  # see the startup zero check under main.py
```

Completed flow:
1. Camera and arm initialization (startup zero check), move to ready position
2. Real-time camera preview + YOLO detection of the two objects, with the reach zones drawn on the view
3. Press `G` to freeze the frame, compute the arm poses via hand-eye transformation and check each one with IK
4. Arm grasps the object and lifts it; the wrist camera checks it is really held
5. Arm moves above the destination, releases the object and returns to the ready position
6. Press `Q` to exit; the gripper is released and the arm returns to zero

| Key | Action |
| --- | --- |
| `P` / `D` | Type a new pick object / destination in the window (`Enter` applies, `Esc` cancels) |
| `G` | Capture and execute pick and place |
| `R` | Resume the live preview after a capture |
| `Q` / `Esc` | Release the gripper, home and exit |

Two sliders in the window set the detection confidence threshold and the release clearance.

**Release height.** The object is released at the destination's top plus the clearance, not at a fixed height. The top is measured from depth over the destination's segmentation mask (a cup's rim; about 0 for a spot on the table) and smoothed over recent frames. The panel in the top-right corner shows the measured top.

**Reach zones.** Green = the arm can pick here; blue stripes = it can release here, at the current release height. Both come from the arm's IK at the ready pose, the pose every pick starts from, so they apply only while the arm is there. The pick zone is computed once at startup (a few seconds). The release zone is recomputed in the background when the release height changes by at least 3 cm for 1.5 s. A higher release point leaves a smaller release zone, and above some height the arm cannot reach it anywhere in view. Before moving, the pregrasp, grasp and release poses are each checked with IK, so an object that could not be placed is never picked up.

**Held check.** After lifting, the wrist camera looks into the gap between the jaws: a held object is close to the camera, an empty gripper shows the table far behind. This catches light or size-limited grasps the gripper motor cannot feel. The last check's images are saved to `Log/held_check_latest.jpg` and `Log/held_check_depth.jpg` for tuning. Disable it with `grasp_pipeline.held_check.enabled: false`.


### 4. `scripts/ordinary_grasp_pipeline.py` — Simplified Grasp Testing

Does not depend on the robotic arm; only verifies OBB grasp pose estimation and visualization effects, suitable for debugging the perception module.

### 5. `scripts/graspnet_camera_demo.py` — GraspNet Camera Estimation Demo

Does not connect to the robotic arm; only runs GraspNet 6D grasp pose estimation using the RGB-D camera. The script keeps a live camera preview, uses YOLO detection boxes to select the target area, then filters feasible grasp candidates from GraspNet full-scene candidates within the target bbox. Press `G` or `Space` to run inference on the current frame, press `R` to resume live preview, press `Q` or `Esc` to exit; after inference, you can view the point cloud and grasp candidates via Open3D.

```bash
python scripts/graspnet_camera_demo.py
```

### 6. `scripts/grasp.py` — GraspNet Robotic Arm Grasping Program

Connects GraspNet estimation results to the robotic arm execution flow based on `graspnet_camera_demo.py`: YOLO selects the target, GraspNet outputs 6D grasp pose, hand-eye calibration transforms it to the robot base frame, then checks IK reachability and executes pre-grasp, grasp, and retreat motions. For debugging, it is recommended to first use `--dry-run` to only print target pose and candidate filtering results.

```bash
python scripts/grasp.py --dry-run
python scripts/grasp.py --target-class "light blue coffee cup"
```

### 7. `scripts/object_detection.py` — Basic Detection Demo

Pure YOLO detection demonstration with real-time display of detection boxes and confidence scores, no grasping logic.

---

## default.yaml Parameter Description

#### 1. Camera and Calibration Configuration (`camera` & `calibration`)

| Parameter | Type / Options | Meaning and Description |
| :--- | :--- | :--- |
| `camera.type` | `realsense_d435i`<br/>`realsense_d405`<br/>`orbbec_gemini2` | **Camera Type**: Specifies the camera hardware connected to the current system. |
| `camera.serial` | `string` / `null` | **Device Serial Number**: Specifies the device SN number. With `null`, a RealSense camera is chosen by the model in `camera.type` (e.g. the D405 when a D405 and a D435i are both connected); with only one camera connected, that camera is used. |
| `calibration.aruco.marker_length_m` | `float` | **ArUco Marker Size**: The actual physical side length of the ArUco calibration marker used for hand-eye calibration, in **meters (m)**. |
| `calibration.hand_eye_compensation_m` | `array` | **Hand-Eye Calibration Translation Compensation**: XYZ manual translation compensation (format `[X, Y, Z]`) executed in the **robot base frame** after hand-eye calibration is complete, in **meters (m)**. If all three values are `0.0`, the compensation matrix is the identity matrix. |

---

#### 2. Object Detection Configuration (`detection`)

| Parameter | Type | Meaning and Description |
| :--- | :--- | :--- |
| `detection.conf_threshold` | `float` | **YOLO Detection Confidence Threshold**: Detection boxes with scores below this value will be filtered out. |
| `detection.iou_threshold` | `float` | **YOLO NMS IoU Threshold**: The Intersection over Union (IoU) threshold used in Non-Maximum Suppression (NMS) to filter overlapping boxes. |

---

#### 3. Robot and Gripper Configuration (`robot`)

| Parameter | Type / Options | Meaning and Description |
| :--- | :--- | :--- |
| `robot.repo_root` | `string` / `null` | **Repository Root Directory**: Path to the `reBotArm_control_py` repository. When `null`, defaults to the internal relative path `sdk/reBotArm_control_py`. |
| `robot.zero_check_tol_rad` | `float` | **Zero Check Tolerance**: At startup, every motor must read within this many radians of zero (arm folded at rest, gripper closed). See the startup zero check under `scripts/main.py`. |
| `robot.ready_pose` | `array` | **Ready Pose**: The ready position the system arm moves to on startup. After each grasping task completes, the arm also automatically returns to this position. |
| `robot.gripper.dm`<br/>`robot.gripper.rs` | struct object | **Gripper Hardware Parameters**: The system automatically selects and applies one of these two parameter groups based on the current actual hardware configuration in the SDK. |

#### Gripper Internal Core Parameter Description

For sub-parameters inside `robot.gripper.dm` or `robot.gripper.rs`:

* **`angle_open`**, **`close_torque`**, **`default_force`**: Correspond to open angle, close torque, and default control force respectively; all must be填写**positive numbers**.
* **`counterclockwise`**: Boolean value. Indicates the motor rotation direction used when closing (whether counterclockwise). The code automatically derives the signs of the open angle and close torque based on this logic.
* **`tau_max`**: Torque upper limit.

`robot.gripper.size_aware` applies to both arm versions and bounds the jaws by the object width measured by vision:

* **`enabled`**: `false` restores the original behaviour (open fully, close until contact).
* **`open_margin_m`**: Open this much wider than the object before approaching it.
* **`squeeze_m`**: Never close more than this below the measured width.
* **`fit_clearance_m`**: Skip objects wider than the maximum jaw opening minus this.
* **`contact_speed_ratio`**: When the jaws reach the width limit, they count as touching the object if they were moving slower than this fraction of their peak closing speed.

:::tip
Note: For other advanced gripper control behavior parameters, refer to and define them in the `drivers/robot/grasp_driver.py` file.
:::

---

#### 4. Grasp Pipeline and GraspNet Configuration (`grasp_pipeline` & `graspnet`)

| Parameter | Type | Meaning and Description |
| :--- | :--- | :--- |
| `grasp_pipeline.infer_every_live` | `int` | **Inference Frame Interval**: During real-time video preview, run object detection every N frames to effectively reduce CPU/GPU real-time computation load. |
| `grasp_pipeline.grasp.depth_quantile` | `float` | **Depth Quantile**: The depth calculation quantile used by the short-axis grasping pipeline. A larger value typically results in a deeper grasp point. |
| `grasp_pipeline.grasp.pregrasp_offset_m` | `float` | **Pre-grasp Position Offset**: The distance to retreat along the end effector feed direction relative to the final target grasp position, in **meters (m)**. |
| `grasp_pipeline.grasp.insertion_depth_m` | `float` | **Insertion Depth**: The additional forward push or insertion depth along the feed direction when GraspNet executes grasping, in **meters (m)**. |
| `grasp_pipeline.held_check.enabled` | `bool` | **Held Check**: After lifting, check with the wrist camera that something is between the jaws (`scripts/set.py`). |
| `grasp_pipeline.held_check.near_margin_m` | `float` | **Held Check Margin**: Depth within this distance beyond the gripper point counts as a held object, in **meters (m)**. |
| `place_task.pick`<br/>`place_task.place` | `string` | **Pick-and-Place Labels**: Starting pick object and destination for `scripts/set.py`; both can be changed in its window. |
| `place_task.release_clearance_m` | `float` | **Release Clearance**: Height above the destination's measured top at which the object is released, in **meters (m)**. |
| `grasp_pipeline.grasp.min_base_z_m` | `float` | **Minimum Grasp Height Limit**: The minimum allowable grasp Z-axis height in the **robot base frame**, in **meters (m)** (used as a底层防碰撞safety boundary). |
| `graspnet` | struct config | **GraspNet Runtime Parameters**: All sub-parameters under this configuration item are loaded when running `scripts/graspnet_camera_demo.py` and `scripts/grasp.py`. |

### Model Selection Library

The YOLO model loads from the `rebot_grasp/models/` directory; if the model file does not exist, Ultralytics usually attempts to download it automatically.

Common models:

| Model | Description |
| --- | --- |
| `yoloe-26l-seg.pt` | Open vocabulary + segmentation, current default |
| `yoloe-26s-seg.pt` | Lighter weight, faster |
| `yolov8n-seg.pt` | Closed-category segmentation, small model |
| `yolov8s-seg.pt` | Closed-category segmentation, higher precision |

When the model name contains `world` / `yoloe` and `yolo.use_world=true`, the program calls `model.set_classes(custom_classes)` to inject `yolo.custom_classes` as open-vocabulary classes. Regular `yolov8*-seg.pt` models ignore this set of open-vocabulary classes.

---

## ❓ FAQ

### 1. `ModuleNotFoundError: No module named 'motorbridge'`

This usually means the robotic arm SDK dependencies are not installed in the current Python environment. Confirm that the project environment is activated and resynchronize the environment and install the robotic arm SDK:

```bash
conda activate rebotarm
conda env update -n rebotarm -f environment.yml
cd sdk/reBotArm_control_py && pip install -e .
```

### 2. Pressing `G` does not execute grasping

Common causes:

- `hand_eye.npz` does not exist
- The hand-eye calibration mode is not `eye_in_hand`
- The current target pose is not IK reachable

It is recommended to first use dry-run mode to verify perception results and target pose:

```bash
python scripts/main.py --dry-run
```

### 3. Grasp point depth is unstable

Priority can be given to checking and adjusting:

- `grasp_pipeline.grasp.depth_quantile`
- The installation height of the camera relative to the target workspace
- Reflectivity of the target surface

### 4. GraspNet reports `pointnet2_utils` cannot be imported from `pointnet2`

This is usually because the local CUDA extension under `sdk/graspnet-baseline/pointnet2` was not properly built and installed in the current conda environment, or Python is resolving to a wrong `pointnet2` package. It is recommended to confirm the project environment is activated and rebuild and install both `pointnet2` and `knn` in the same environment:

```bash
conda activate rebotarm
cd sdk/graspnet-baseline/pointnet2
pip install . --no-build-isolation

cd ../knn
pip install . --no-build-isolation
```

Verify:

```bash
python -c "from pointnet2 import pointnet2_utils; print('Submodule import works')"
```

### 5. CUDA architecture incompatibility when running GraspNet on the current graphics card

If you see `no kernel image is available for execution on the device` or PyTorch reports that the current GPU's CUDA capability is unsupported, this usually means the current PyTorch wheel does not include CUDA kernels for that graphics card architecture. It is recommended to install a PyTorch version that supports the current CUDA/graphics card architecture, then rebuild the GraspNet local CUDA extensions.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

cd sdk/graspnet-baseline/pointnet2
pip install . --no-build-isolation

cd ../knn
pip install . --no-build-isolation
```

If you need to manually specify the build architecture, set `TORCH_CUDA_ARCH_LIST` before rebuilding, with the specific value confirmed according to your current graphics card architecture and PyTorch/CUDA version.

### 6. GraspNet inference reports `RuntimeError: CPU not supported`

The sampling operators in `pointnet2` only support CUDA tensors. Confirm that CUDA is available, the GraspNet network and input point cloud are on GPU, and `pointnet2` / `knn` were built against the current environment and PyTorch version.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is `False`, you need to fix the CUDA / PyTorch installation first; if the output is `True` but the error persists, it is recommended to rebuild `pointnet2` and `knn`.

---

## 📄 References

- [reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py) — Robotic arm control library
- [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) — reBot robotic arm open source project
- [Orbbec Gemini 2 Product Page](https://www.orbbec.com.cn/index/Product/info.html?cate=38&id=51)
- [Orbbec SDK v2](https://github.com/orbbec/OrbbecSDK_v2)
- [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)
- [RealSense SDK](https://github.com/realsenseai/librealsense)
- [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)

---

## ☎ Contact Us

- **Technical Support**: [Submit an Issue](https://github.com/Seeed-Projects/reBot-DevArm-Grasp/issues)

---

<p align="center">
  <strong>🌟 If this project is helpful to you, please give it a Star!</strong>
</p>