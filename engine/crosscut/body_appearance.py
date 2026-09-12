"""Digest-bound, source-local body appearance, never named-person identity.

OSNet-AIN weights are an explicit external asset. Appearance vectors stay in
memory. This encoder neither downloads assets nor turns similarity into an
identity probability; source-local tracks must corroborate it with face and
temporal evidence. The architecture and weights come from the MIT-licensed
official deep-person-reid project.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from threading import Lock
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

OSNET_AIN_MSMT17_SHA256 = '8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f'


class BodyAppearanceEncoder:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._model = None
        self._lock = Lock()
        self.inference_image_count = 0

    def fingerprint(self) -> dict:
        return {'contract': 'osnet_ain_source_local_body_appearance_v1',
            'model_sha256': OSNET_AIN_MSMT17_SHA256,
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'architecture_sha256': hashlib.sha256(Path(__file__).with_name('_osnet_ain_network.py').read_bytes()).hexdigest(),
            'preprocessing': 'PIL_RGB_bilinear_256x128_imagenet_mean_std_v1',
            'embeddings_persisted': False, 'identity_name_inferred': False}

    def _load(self):
        import torch
        if self._model is not None:
            return self._model
        digest = hashlib.sha256()
        with self.model_path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != OSNET_AIN_MSMT17_SHA256:
            raise ValueError('body appearance model digest mismatch')
        from ._osnet_ain_network import osnet_ain_x1_0
        state = torch.load(self.model_path, weights_only=True, map_location='cpu')
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        model = osnet_ain_x1_0(num_classes=state['classifier.weight'].shape[0])
        model.load_state_dict(state, strict=True)
        self._model = model.eval()
        return self._model

    def describe(self, frame: np.ndarray, boxes: Sequence[tuple[float, float, float, float]]) -> tuple[tuple[float, ...], ...]:
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size):
            raise ValueError('body appearance requires actual uint8 BGR pixels')
        if len(boxes) > 32:
            raise ValueError('body appearance batch exceeds 32 observed people')
        crops = []
        h, w = frame.shape[:2]
        for box in boxes:
            if (len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) for v in box)
                    or min(box[:2]) < 0 or min(box[2:]) <= 0
                    or box[0]+box[2] > 1.000001 or box[1]+box[3] > 1.000001):
                raise ValueError('body appearance box must fit the observed image')
            x, y, bw, bh = box
            crop = frame[max(0, int(y*h)):min(h, math.ceil((y+bh)*h)),
                         max(0, int(x*w)):min(w, math.ceil((x+bw)*w))]
            if not crop.size:
                raise ValueError('body appearance crop is empty')
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((128, 256), Image.Resampling.BILINEAR))
        if not crops:
            return ()
        import torch
        # This also supports the installed torch build whose NumPy bridge
        # predates NumPy 2. No global dependency downgrade is necessary.
        mean = torch.tensor((.485, .456, .406)).reshape(3, 1, 1)
        std = torch.tensor((.229, .224, .225)).reshape(3, 1, 1)
        values = []
        for crop in crops:
            tensor = torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8).reshape(256, 128, 3)
            tensor = tensor.permute(2, 0, 1).to(dtype=torch.float32).div(255)
            values.append((tensor-mean)/std)
        with self._lock, torch.inference_mode():
            model = self._load()
            rows = []
            for start in range(0, len(values), 8):
                features = model(torch.stack(values[start:start+8]))
                if features.ndim != 2 or features.shape[1] != 512 or not torch.isfinite(features).all():
                    raise RuntimeError('body appearance model output is invalid')
                rows.extend(torch.nn.functional.normalize(features, p=2, dim=1).tolist())
            self.inference_image_count += len(values)
        return tuple(tuple(float(v) for v in row) for row in rows)
