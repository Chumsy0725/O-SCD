# Changes in Real Time: Online Scene Change Detection with Multi-View Fusion
*Chamuditha Jayanga, Jason Lai, Lloyd Windrim, Donald Dansereau, Niko Suenderhauf, Dimity Miller*

[![Static Badge](https://img.shields.io/badge/Project%20Page-%23ecf0f1?logo=homepage&logoColor=%23222222&link=https%3A%2F%2Fchumsy0725.github.io%2FMV-3DCD%2F)](https://chumsy0725.github.io/O-SCD/)

*Abstract*:  Online Scene Change Detection (SCD) is an extremely challenging problem that requires an agent to detect relevant changes on the fly while observing the scene from unconstrained viewpoints. Existing online SCD methods are significantly less accurate than offline approaches. We present the first online SCD approach that is pose-agnostic, label-free, and ensures multi-view consistency, while operating at over 10 FPS and achieving new state-of-the-art performance, surpassing even the best offline approaches.
    Our method introduces a new self-supervised fusion loss to infer scene changes from multiple cues and observations, PnP-based fast pose estimation against the reference scene, and a fast change-guided update strategy for the 3D Gaussian Splatting scene representation. Extensive experiments on complex real-world datasets demonstrate that our approach outperforms both online and offline baselines.

## BibTeX
```shell


@inproceedings{galappaththige2026online,
  title={Changes in Real Time: Online Scene Change Detection with Multi-View Fusion},
  author={Galappaththige, Chamuditha Jayanga and Lai, Jason and Windrim, Lloyd and Dansereau, Donald and Sunderhauf, Niko and Miller, Dimity},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2026}
}
```
## Environment Setup

We use `Conda` for environment and package management. 

```shell
conda create -n oscd python=3.12 -y
conda activate oscd

pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
pip install cupy-cuda12x
pip install -r requirements.txt

```

I have incorporated the released codebase with [FastGS](https://github.com/fastgs/FastGS). This would give you slightly better and faster performance than reported. 

## Running Experiments

To run the SCD experiments for all 20 instances, you can use the `run_oscd.sh` script
```shell
bash run_oscd.sh
```

To run the Scene Update for all 20 instances, you can use the `run_update.sh` script
```shell
bash run_oscd.sh
```

To initialize the viewer for any given scene (after running Scene Update/SCD), use the `run_viewer.sh` script
```shell
bash run_viewer.sh "Scene" "Instance"
```

### Acknowledgement

Our code is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [on-the-fly-nvs](https://github.com/graphdeco-inria/on-the-fly-nvs), [FastGS](https://github.com/fastgs/FastGS) and [SAMv2](https://github.com/facebookresearch/sam2). We sincerely thank the authors for open-sourcing their codebase. 

### Funding Acknowledgement

This work was supported by the ARC Research Hub in Intelligent Robotic Systems for Real-Time Asset Management (ARIAM) (IH210100030) and Abyss Solutions. C.J., N.S., and D.M. also acknowledge ongoing support from the QUT Centre for Robotics.



