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

import argparse
import queue
import time
from threading import Thread

import cv2
import torch
from torch import Tensor


class StreamDataset:
    def __init__(self, video_url: str, downsampling: float, retry_delay: float = 1.0):
        """
        Args:
            video_url (str): video stream URL.
            retry_delay (int): Delay in seconds between retries.
        """
        self.video_url = video_url
        self.downsampling = downsampling

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.retry_delay = retry_delay
        self.cap = None

        # Thread to get frames from the video stream
        self.capture_thd = Thread(target=self._capture_frames, daemon=True)
        self.capture_thd.start()

        self.num_frames = 0
    
    def _connect(self):
        if self.cap is not None:
            return

        cap = cv2.VideoCapture(self.video_url)
        if not cap.isOpened():
            print(f"Failed to open video stream: {self.video_url}")
            return

        print("Connected to camera stream.")
        self.cap = cap

    def _capture_frames(self) -> None:
        while self.running:
            if self.cap is None:
                self._connect() 
                time.sleep(self.retry_delay)
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from stream.")
                self.cap.release()
                self.cap = None
                continue

            if not self.frame_queue.empty():
                self.frame_queue.get()  # Discard the older frame
            self.frame_queue.put(frame)

    def getnext(self) -> tuple[Tensor, dict]:
        frame = self.frame_queue.get(block=True)
        self.num_frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.downsampling > 0.0 and self.downsampling != 1.0: 
            frame = cv2.resize(
                frame,
                (0, 0),
                fx=1 / self.downsampling,
                fy=1 / self.downsampling,
                interpolation=cv2.INTER_AREA,
            )
        image = torch.from_numpy(frame).permute(2, 0, 1).cuda().float() / 255.0
        info = {"is_test": False}
        return image, info

    def get_image_size(self):
        frame = self.getnext()[0]
        self.num_frames -= 1
        return frame.shape[-2], frame.shape[-1]

    def stop(self) -> None:
        self.running = False
        self.cap.release()
        self.capture_thd.join()
    
    def __len__(self):
        # Arbitrary large number as we don't know the length of a stream
        return 100_000_000  

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Dataset")
    parser.add_argument("-s", "--source_path", type=str, help="video stream URL")
    parser.add_argument("--downsampling", type=float, default=1.5)
    args = parser.parse_args()

    stream = StreamDataset(args.source_path, args.downsampling)
    try:
        while True:
            image, info = stream.getnext()
            # Example processing
            cv2.imshow("Stream", image.permute(1, 2, 0).cpu().numpy()[..., ::-1])
            cv2.waitKey(1)
    except KeyboardInterrupt:
        stream.stop()
