# Changes in Real Time: Online Scene Change Detection with Multi-View Fusion

*Chamuditha Jayanga, Jason Lai, Lloyd Windrim, Donald Dansereau, Niko Suenderhauf, Dimity Miller*

**CVPR 2026 (Highlight)**

[![Project Page](https://img.shields.io/badge/Project%20Page-O--SCD-1f77b4)](https://chumsy0725.github.io/O-SCD/)
[![Paper](https://img.shields.io/badge/Paper-CVPR%202026-4b5563)](https://openaccess.thecvf.com/content/CVPR2026/html/Galappaththige_Changes_in_Real_Time_Online_Scene_Change_Detection_with_Multi-View_CVPR_2026_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2511.12370-b31b1b)](https://arxiv.org/abs/2511.12370)
[![Dataset](https://img.shields.io/badge/Dataset-PASLCD-ffd21e)](https://huggingface.co/datasets/ChamudithaJay/PASLCD)

This directory holds the self-contained code for O-SCD. It is part of the [O-SCD line](../README.md); see the repository README for the other methods.

<p align="center">
<img src="assets/oscd.gif" alt="O-SCD detecting changes online" width="800"> </p>

*Abstract*: Online Scene Change Detection (SCD) is an extremely challenging problem that requires an agent to detect relevant changes on the fly while observing the scene from unconstrained viewpoints. Existing online SCD methods are significantly less accurate than offline approaches. We present the first online SCD approach that is pose-agnostic, label-free, and ensures multi-view consistency, while operating at over 10 FPS and achieving new state-of-the-art performance, surpassing even the best offline approaches.
Our method introduces a new self-supervised fusion loss to infer scene changes from multiple cues and observations, PnP-based fast pose estimation against the reference scene, and a fast change-guided update strategy for the 3D Gaussian Splatting scene representation. Extensive experiments on complex real-world datasets demonstrate that our approach outperforms both online and offline baselines.

## Environment Setup

We use `Conda` for environment and package management. All commands below are run from this directory.

```shell
git clone https://github.com/Chumsy0725/O-SCD.git
cd O-SCD/O-SCD

conda create -n oscd python=3.12 -y
conda activate oscd

pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
pip install cupy-cuda12x
pip install -r requirements.txt
pip install viser  # only needed for the interactive viewer
```

I have incorporated the released codebase with [FastGS](https://github.com/fastgs/FastGS). This would give you slightly better and faster performance than reported.

## Data Preparation

Download the PASLCD (online) dataset from [HuggingFace](https://huggingface.co/datasets/ChamudithaJay/PASLCD/blob/main/PASLCD_online.zip) and unzip it into the `data` directory. The run scripts read scenes from `data/PASLCD/Instance_*/<Scene>`; edit the data path at the top of each script if yours differs.

## Running Experiments

To run the SCD experiments for all 20 instances, you can use the `run_oscd.sh` script.
```shell
bash run_oscd.sh
```

To run the Scene Update for all 20 instances, you can use the `run_update.sh` script.
```shell
bash run_update.sh
```

To initialize the viewer for any given scene (after running Scene Update/SCD), use the `run_viewer.sh` script.
```shell
bash run_viewer.sh "Scene" "Instance"
```

<!-- ## BibTeX

```bibtex
@inproceedings{galappaththige2026online,
  title     = {Changes in Real Time: Online Scene Change Detection with Multi-View Fusion},
  author    = {Galappaththige, Chamuditha Jayanga and Lai, Jason and Windrim, Lloyd and Dansereau, Donald and Suenderhauf, Niko and Miller, Dimity},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {32246--32256},
  year      = {2026}
}
``` -->

## License

This code is released for non-commercial academic research and evaluation only. Third-party components retain their own licenses.
