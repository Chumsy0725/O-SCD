#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations
from argparse import Namespace
import torch
import torch.nn.functional as F

from poses.feature_detector import DescribedKeypoints
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from utils.utils_fly import sample, sixD2mtx, make_torch_sampler, depth2points
from dataloaders.read_write_model import Camera, BaseImage, rotmat2qvec


class Keyframe:
    """
    A keyframe in the scene, containing the image, camera parameters, and other information used for optimization.
    """
    def __init__(
        self,
        image: torch.Tensor,
        info: dict,
        desc_kpts: DescribedKeypoints,
        Rt: torch.Tensor,
        index: int,
        f: torch.Tensor,
        feat_extractor: DenseExtractor,
        triangulator: Triangulator,
        args: Namespace,
        inference_mode: bool = False,
    ):
        self.image_pyr = [image]
        if not inference_mode:
            self.feat_map = feat_extractor(image)
            self.width = image.shape[2]
            self.height = image.shape[1]
            self.centre = torch.tensor(
                [(self.width - 1) / 2, (self.height - 1) / 2], device="cuda"
            )
            self.f = f
            self.triangulator = triangulator
            for _ in range(args.pyr_levels - 1):
                self.image_pyr.append(F.avg_pool2d(self.image_pyr[-1], 2))
            self.mask_pyr = info.pop("mask", None)
            self.pyr_lvl = args.pyr_levels - 1
            if self.mask_pyr is not None:
                self.mask_pyr = [self.mask_pyr.cuda()]
                for _ in range(args.pyr_levels - 1):
                    self.mask_pyr.append(F.avg_pool2d(self.mask_pyr[-1], 2))
                for i in range(len(self.mask_pyr)):
                    self.mask_pyr[i] = self.mask_pyr[i] > (1 - 1e-6)

        self.index = index

        self.idepth_pyr = None

        self.latest_invdepth = None
        self.desc_kpts = desc_kpts
        self.info = info
        self.is_test = info["is_test"]

        self.rW2C = Rt[:3, :2].detach().clone()
        self.tW2C = Rt[:3, 3].detach().clone()

        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def to(self, device: str, only_train=False):
        if self.device.type == device:
            return
        for i in range(len(self.image_pyr)):
            self.image_pyr[i] = self.image_pyr[i].to(device)
            if self.idepth_pyr is not None:
                self.idepth_pyr[i] = self.idepth_pyr[i].to(device)
            if self.mask_pyr is not None:
                self.mask_pyr[i] = self.mask_pyr[i].to(device)
        if not only_train:
            self.feat_map = self.feat_map.to(device)
            self.mono_idepth = self.mono_idepth.to(device)
            if self.latest_invdepth is not None:
                self.latest_invdepth = self.latest_invdepth.to(device)

    @property
    def device(self):
        return self.image_pyr[0].device

    def get_R(self):
        return sixD2mtx(self.rW2C)

    def get_t(self):
        return self.tW2C

    def get_Rt(self):
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.get_t()
        return Rt

    def set_Rt(self, Rt: torch.Tensor):
        self.rW2C.data.copy_(Rt[:3, :2])
        self.tW2C.data.copy_(Rt[:3, 3])

        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def get_centre(self, approx=False):
        if approx:
            return self.approx_centre
        else:
            return -self.get_R().T @ self.get_t()

    @torch.no_grad()
    def update_3dpts(self, all_keyframes: list[Keyframe]):
        """
        Assign a 3D point to each keypoint in the keyframe based on triangulation and the latest rendered depth. 
        """
        unload_desc_kpts = self.desc_kpts.kpts.device.type == "cpu"
        if unload_desc_kpts:
            self.desc_kpts.to("cuda")


        ## Triangulation
        # Select keyframes to triangulate with based on their locations
        uv, uvs_others, chosen_kfs_ids = self.triangulator.prepare_matches(
            self.desc_kpts
        )
        Rts_others = torch.stack(
            [all_keyframes[index].get_Rt() for i, index in enumerate(chosen_kfs_ids)],
            dim=0,
        )
        if len(Rts_others < self.triangulator.n_cams):
            Rts_others = torch.cat(
                [
                    Rts_others,
                    torch.eye(4, device="cuda")[None].repeat(
                        self.triangulator.n_cams - len(Rts_others), 1, 1
                    ),
                ],
                dim=0,
            )

        # Run the triangulator and update the 3D points
        new_pts, depth, best_dis, valid_matches = self.triangulator(
            uv, uvs_others, self.get_Rt(), Rts_others, self.f, self.centre
        )
        self.desc_kpts.update_3D_pts(
            new_pts[valid_matches], depth[valid_matches], 1, valid_matches
        )

        if unload_desc_kpts:
            self.desc_kpts.to("cpu")

    def to_json(self):
        info = {
            "is_test": self.info["is_test"],
        }
        if "name" in self.info:
            info["name"] = self.info["name"]
        if "Rt" in self.info:
            info["gt_Rt"] = self.info["Rt"].cpu().numpy().tolist()

        return {
            "info": info,
            "Rt": self.get_Rt().detach().cpu().numpy().tolist(),
            "f": self.f.item(),
        }

    @classmethod
    def from_json(cls, config, index, height, width):
        if "gt_Rt" in config["info"]:
            config["info"]["Rt"] = torch.tensor(config["info"]["gt_Rt"]).cuda()
        keyframe = cls(
            image=None,
            info=config["info"],
            desc_kpts=None,
            Rt=torch.tensor(config["Rt"]).cuda(),
            index=index,
            f=None, 
            feat_extractor=None,
            depth_estimator=None,
            triangulator=None,
            args=None,
            inference_mode=True,
        )
        keyframe.height = height
        keyframe.width = width
        keyframe.centre = torch.tensor(
            [(width - 1) / 2, (height - 1) / 2], device="cuda"
        )
        return keyframe

    def to_colmap(self, id):
        """
        Convert the keyframe to a colmap camera and image.
        """
        # first param of params is focal length in pixels
        camera = Camera(
            id=id,
            model="SIMPLE_PINHOLE",
            width=self.width,
            height=self.height,
            params=[self.f.item(), self.centre[0].item(), self.centre[1].item()],
        )

        image = BaseImage(
            id=id,
            name=self.info.get("name", str(id)),
            camera_id=id,
            qvec=-rotmat2qvec(self.get_R().cpu().detach().numpy()),
            tvec=self.get_t().flatten().cpu().detach().numpy(),
            xys=[],
            point3D_ids=[],
        )

        return camera, image
