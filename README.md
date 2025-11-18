# Changes in Real Time: Online Scene Change Detection with Multi-View Fusion
*Chamuditha Jayanga, Jason Lai, Lloyd Windrim, Donald Dansereau, Niko Suenderhauf, Dimity Miller*

[![Static Badge](https://img.shields.io/badge/Paper-%23ecf0f1?logo=arxiv&logoColor=%23B31B1B&link=https%3A%2F%2Fchumsy0725.github.io%2FMV-3DCD%2F)](https://arxiv.org/abs/2511.12370/) 

![alt text](./assets/logos.png)

*Abstract*:  Online Scene Change Detection (SCD) is an extremely challenging problem that requires an agent to detect relevant changes on the fly while observing the scene from unconstrained viewpoints. Existing online SCD methods are significantly less accurate than offline approaches. We present the first online SCD approach that is pose-agnostic, label-free, and ensures multi-view consistency, while operating at over 10 FPS and achieving new state-of-the-art performance, surpassing even the best offline approaches.
    Our method introduces a new self-supervised fusion loss to infer scene changes from multiple cues and observations, PnP-based fast pose estimation against the reference scene, and a fast change-guided update strategy for the 3D Gaussian Splatting scene representation. Extensive experiments on complex real-world datasets demonstrate that our approach outperforms both online and offline baselines.
