# Validation

코드의 정확성과 영상의 편집 품질은 따로 확인합니다.

공개본 준비 시 확인한 결과(2026-11-13):

- 아래 핵심 회귀 테스트 215개 통과
- Python 엔진 전체 소스 문법 검사와 CLI 도움말 실행 통과
- macOS `CrossCutMac` Debug 빌드와 Python 엔진 번들링 통과 (`CODE_SIGNING_ALLOWED=NO`)

이 확인 과정에서는 영상 수집·학습·실제 공연 영상 렌더를 실행하지 않았습니다.

## 자동 검증

공개 저장소의 CI는 실제 공연 영상을 다운로드하거나 모델을 학습하지 않고, 합성 입력을 사용하는 핵심 회귀 테스트를 실행합니다. 시간 정렬, 입력 검증, 타임라인 선택과 렌더 명령 구성을 확인하는 범위입니다.

```bash
python -m unittest -q \
  engine.tests.test_audio \
  engine.tests.test_request \
  engine.tests.test_timeline \
  engine.tests.test_visual_score \
  engine.tests.test_render \
  engine.tests.test_render_frame_schedule \
  engine.tests.test_boundary_affine_match
```

전체 테스트 디렉터리에는 추가 런타임·학습 환경·외부 모델이 필요한 연구용 테스트도 포함되어 있습니다. 위 명령은 기본 회귀 범위이며, 모든 선택 기능을 검증하는 명령은 아닙니다.

## 문제 해결 사례 검증

README의 세 가지 사례에 대응하는 기존 테스트 모듈을 2026-11-13에 실행하여 **24개 통과**를 확인했습니다. 합성 마커의 실제 FFmpeg 출력 좌표, 4초 합성 영상의 인코딩 패킷 재사용, 얼굴 색 보정의 적용 범위를 검증합니다. 실제 공연 영상의 추가 렌더나 학습은 실행하지 않았습니다.

```bash
python -m pip install pytest
python -m pytest -q \
  engine/tests/test_crosscut2_aspect_viewport.py \
  engine/tests/test_crosscut2_encoded_programme_reuse.py \
  engine/tests/test_crosscut2_member_face_tone.py
```

## 영상 평가

단체 영상과 개인 직캠을 각각 정상 속도로 재생하며 다음 항목을 확인합니다.

- 컷 앞뒤에서 같은 동작이 자연스럽게 이어지는가
- 얼굴 크기·위치와 전체 신체 구도가 갑자기 바뀌지 않는가
- 음악과 안무의 강조 지점에서 전환이 일어나는가
- 샷 길이가 지나치게 짧거나 길지 않은가
- 크롭·확대·회전과 디졸브가 답답함·어지러움·색 변화를 만들지 않는가
- 해상도·프레임·오디오의 일관성이 유지되는가

모델 점수나 테스트 통과만으로 영상 품질을 판정하지 않습니다. README의 대표 시연은 기존 렌더의 발췌입니다. 공연 원본과 전체 렌더는 저장소에 포함하지 않으므로, 전체 영상 비교에는 별도 소스가 필요합니다.
