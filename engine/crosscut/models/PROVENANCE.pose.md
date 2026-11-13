# OpenCV Zoo pose and silhouette assets

These files are bundled for local, on-device visual analysis. They were
downloaded without modification from the official `opencv/opencv_zoo`
repository at commit `47534e27c9851bb1128ccc0102f1145e27f23f98`:

- `person_detection_mediapipe_2023mar.onnx`
  - upstream: `models/person_detection_mediapipe/`
  - SHA-256: `47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f`
  - license: Apache License 2.0, see `LICENSE.opencv-zoo-mediapipe-pose`
- `pose_estimation_mediapipe_2023mar.onnx`
  - upstream: `models/pose_estimation_mediapipe/`
  - SHA-256: `9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f`
  - license: Apache License 2.0, see `LICENSE.opencv-zoo-mediapipe-pose`
- `human_segmentation_pphumanseg_2023mar.onnx`
  - upstream: `models/human_segmentation_pphumanseg/`
  - SHA-256: `552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24`
  - license: Apache License 2.0, including the PaddlePaddle copyright
    notice in `LICENSE.opencv-zoo-pphumanseg`
- `object_detection_nanodet_2022nov.onnx`
  - upstream: `models/object_detection_nanodet/`
  - model variant: NanoDet-m-plus-1.5x at 416x416
  - use: non-identifying full-stage formation person boxes only
  - SHA-256: `4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186`
  - license: Apache License 2.0, see
    `LICENSE.opencv-zoo-mediapipe-pose` (the upstream license text is byte
    identical)

Official sources:

- https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/person_detection_mediapipe
- https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/pose_estimation_mediapipe
- https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/human_segmentation_pphumanseg
- https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_nanodet

`crosscut/pose.py` contains a compact adaptation of the official Python
pre/post-processing examples under the same Apache-2.0 terms. The editor does
not download models or send frames over the network at runtime.
