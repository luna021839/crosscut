# External models and references

## 포함된 기본 모델

모델 파일과 라이선스는 `engine/crosscut/models/`에 보존되어 있습니다.

| 모델 | 용도 | 동봉 라이선스 |
| --- | --- | --- |
| OpenCV Zoo YuNet | 얼굴 검출과 랜드마크 | `LICENSE.yunet` — MIT |
| OpenCV Zoo SFace | 얼굴 특징의 유사도 | `LICENSE.opencv-zoo-sface` — Apache-2.0 |
| OpenCV Zoo MediaPipe person / pose | 인물 검출과 신체 포즈 | `LICENSE.opencv-zoo-mediapipe-pose` — Apache-2.0 |
| OpenCV Zoo PP-HumanSeg | 인물 영역 | `LICENSE.opencv-zoo-pphumanseg` — Apache-2.0 |
| OpenCV Zoo NanoDet | 객체 검출 | Apache-2.0 — `PROVENANCE.pose.md`에 원문 위치 기록 |

출처: [OpenCV Zoo](https://github.com/opencv/opencv_zoo). 포즈·실루엣 모델의 정확한 상위 버전과 SHA-256은 [모델 출처 문서](../engine/crosscut/models/PROVENANCE.pose.md)에 기록되어 있습니다. SFace의 모델 SHA-256은 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`이며, YuNet은 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`입니다.

선택적 센서·학습 경로에 관련된 OSNet, RTMPose, SCHP, TransNetV2의 고지는 `engine/crosscut/licenses/`에 보존되어 있습니다. 이 파일들이 해당 모델 가중치의 동봉을 의미하지는 않습니다. 추가 모델은 각 배포처의 조건과 설치 방법을 따릅니다.

## 알고리즘 참고

- [PopStage: The Generation of Stage Cross-Editing Video Based on Spatio-Temporal Optimization](https://doi.org/10.1145/3550454.3555467)
- [AutoStageMix: Fully Automated Stage Cross-Editing System Utilizing Facial Features](https://doi.org/10.3390/app15137613)
- [datamarket-tobigs/Cross-Cutting](https://github.com/datamarket-tobigs/Cross-Cutting): 얼굴 랜드마크의 전환 오차를 비교하는 아이디어 참고. 기존 MoviePy/dlib 구현을 동봉하지 않으며, OpenCV 특징을 사용하는 방식으로 별도 구현했습니다.

## 공개 범위

방송 영상·음원, 인물별 얼굴 데이터, 개인 학습 가중치와 로컬 실험 산출물은 포함하지 않습니다. 제3자 구성요소의 라이선스는 원문을 유지합니다. 프로젝트 자체 코드의 별도 재배포 라이선스는 아직 지정하지 않았습니다.
