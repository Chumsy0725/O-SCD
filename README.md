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
