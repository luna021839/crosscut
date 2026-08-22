# CrossCut

IVE 영상만을 위한 macOS 자동 교차편집 앱입니다. Premiere Pro 프로젝트를 만드는 도구가 아니라, 로컬 원본 영상을 직접 분석해 완성 MP4를 렌더합니다.

## 현재 구현된 모드

### 일반 교차편집

- 같은 곡의 음방 영상 2개 이상 입력
- 오디오 파형으로 영상별 시작 시각 자동 정렬
- 박자·프레이즈·방송 카메라 컷 분석
- 화질, 노출, 구도, 인물 크기, 컷 전후 동작·색감 연속성을 비교해 전역 타임라인 선택
- YuNet 5점 얼굴 랜드마크로 눈 위치·얼굴 크기·기울기·간이 머리 방향을 측정
- 컷 전후 3쌍의 연속 프레임 중 가장 어색한 얼굴 차이를 전역 타임라인 비용에 반영
- 단독·클로즈 샷은 MediaPipe 33개 관절 포즈, 군무·와이드 샷은
  PP-HumanSeg 전체 인물 실루엣으로 컷 전후 자세와 대형 연속성 비교
- 얼굴이 명확하면 최대 1.35배 범위에서 크기·위치를 맞추고, 군무에서는 안정적인 전체 인물 실루엣을 최대 1.18배 범위에서 맞춤
- A: 얼굴 크롭과 미래 3프레임 얼굴·머리·신체 검증에 SFace 외형 호환성 검증까지 통과한 경계만 11프레임 정렬 디졸브
- B: 얼굴 또는 전체 실루엣으로 구도를 맞춘 정렬 하드 컷
- C: 얼굴이 모호해도 박자·동작·포즈·실루엣·색감·화질을 전역 최적화한 교차 하드 컷
- D: 바꾸는 것보다 자연스러운 경우 같은 연속 샷 유지
- 기준 영상의 음원을 하나의 연속 트랙으로 유지
- 16:9 H.264/AAC MP4 출력

입력 하드 제한은 2개이지만, 자연스러운 완곡 교차편집에는 여러 무대가 유리합니다. 9개 이상 입력도 처리하며, 품질이나 싱크가 불안한 영상은 억지로 넣지 않고 자동 제외합니다.

하드 컷은 분석을 포기한 안전 폴백이 아닙니다. 실제 무대 교차편집의 중심인
동일 곡 시각의 컷을 B/C 등급으로 계속 최적화하고, 두 얼굴이나 팔이 겹쳐 보일
위험이 없는 A 등급 경계에만 짧은 디졸브를 더합니다. 각 경계의 등급과 선택·강등
사유는 분석 JSON에 기록됩니다.

SFace는 멤버 이름을 알아내거나 카메라 컷 뒤 대상을 다시 찾는 기능이 아닙니다.
이미 YuNet이 한 명으로 명확하게 잡은 두 짧은 얼굴 궤적 사이의 외형 변화가
블렌드 허용 범위인지 A 등급 후보에서만 보수적으로 확인합니다. 모델이나 얼굴
증거가 없거나 일치도가 부족하면 디졸브만 포기하고, 박자·동작·구도를 분석한
B/C 하드 컷은 그대로
사용합니다. 얼굴 임베딩 원본은 분석 JSON에 저장하지 않습니다.

### 특정 멤버 교차직캠

- 안유진, 가을, 레이, 장원영, 리즈, 이서 중 한 명 선택
- 각 원본에서 대상 멤버를 박스로 한 번 지정
- 지정한 멤버의 연속 궤적을 CSRT/KCF/MIL로 안전 추적
- 각 구간 최상 점수와 0.04 이내인 안전 후보 안에서 최대한 많은 소스를 쓰도록 전역 교차 선택
- 멤버를 따라가는 부드러운 가상 카메라 크롭
- 16:9 또는 9:16 H.264/AAC MP4 출력

현재 멤버 모드는 **컷 없는 고정 멤버 직캠**에 맞춰져 있습니다. 카메라 컷, 가림, 대상 손실이 생기면 다른 멤버로 잘못 점프하지 않도록 그 소스의 추적을 중단합니다. 안전한 소스가 없는 구간이 하나라도 있으면 잘못된 멤버 영상을 만드는 대신 전체 렌더를 중단하고 분석 결과를 남깁니다.

## 권장 입력

- 같은 곡과 같은 편곡을 사용한 무대 영상
- 처음부터 끝까지 공연 음원이 들리는 원본
- 워터마크나 자막이 있어도 되지만, 인물이 충분히 크게 보이는 영상
- 일반 모드: 음방 본방·전체 무대 영상 여러 개
- 멤버 모드: 선택 멤버의 무편집·고정 직캠 여러 개

현재 앱에는 영상 검색이나 다운로드 기능이 없습니다. 보유한 로컬 파일을 드래그하거나 파일 선택기로 넣습니다.

신체 분석 모델이 인물을 검출하지 못해도 그 영상을 자동 제외하지 않습니다.
포즈는 기존 경사도 구조 임베딩으로 결정론적으로 폴백하고, 실루엣 결측은
중립으로 처리합니다. 여러 멤버가 비슷한 크기로 보이면 한 명을 임의로 고르지
않고 전체 인물 실루엣을 사용합니다.

## 앱 사용

1. 곡명과 출력 모드를 선택합니다.
2. 같은 곡의 영상들을 추가합니다.
3. 멤버 모드라면 모든 영상에서 대상 멤버 위치를 한 번씩 지정합니다.
4. 저장 위치를 고르고 `교차편집 시작`을 누릅니다.
5. 결과 폴더에서 MP4, 렌더 계획 JSON, 분석 JSON을 확인합니다.

분석 JSON의 `source_index_mapping`은 두 독립 인덱스 도메인과 렌더 계획의
자체 도메인명을 명시합니다.
`request_source_index`는 사용자가 넣은 `request.sources` 원본 순서입니다.
`eligible_source_index`는 싱크 안전 게이트를 통과한
영상만 0부터 다시 센 분석용 순서입니다. `eligible_to_request`와
`request_to_eligible`로 양방향 변환할 수 있습니다. 렌더 계획 JSON은 자체
`sources[]` 배열 순서인 `render_plan_sources_array_index`로
`reference_source`와 `segments[].source_index`를 표현합니다. 이 파이프라인은
`request.sources`를 렌더 계획의 `sources[]`에 순서 변경 없이 그대로 복사하므로,
두 도메인은 1:1로 같으며 이 동치 관계도 `domain_equivalences`에 기록됩니다.
`payload_index_domains`에는 각 인덱스 필드와 인덱스로 쓰이는 배열 위치의
도메인이 함께 기록됩니다.

## 개발 및 실행

완성 앱의 실행 환경은 macOS 14+, Python 3.11+와 해당 Python에서 import 가능한
NumPy 2.0+/OpenCV Contrib 4.10+, 그리고 `PATH`에서 찾을 수 있는 FFmpeg/ffprobe입니다.
Xcode 16+와 XcodeGen은 소스에서 앱을 다시 빌드할 때만 필요합니다. YuNet,
SFace, MediaPipe Pose, PP-HumanSeg ONNX 모델과 라이선스는 앱에 함께 들어갑니다.
포즈·실루엣·얼굴 임베딩도 OpenCV DNN으로 로컬 실행하므로 MediaPipe,
TensorFlow, ONNX Runtime, Premiere Pro, 클라우드 계정은 필요하지 않습니다.

Homebrew 기준으로 미디어 도구를 준비하고 독립 Python 환경을 만드는 예시는
다음과 같습니다.

```bash
brew install ffmpeg python
python3 -m venv engine/.venv
engine/.venv/bin/python -m pip install "numpy>=2.0" "opencv-contrib-python>=4.10"
```

저장소에서 실행하면 `engine/.venv/bin/python3`을 가장 먼저 찾습니다. 다른 위치의
가상환경을 사용할 때는 앱을 시작하는 환경의 `CROSSCUT_PYTHON`에 그
`python3` 절대 경로를 지정할 수 있습니다. `doctor`의 `ready`는 필수 미디어
도구와 Python/NumPy/OpenCV 버전 계약을 만족하고 일반·멤버 두 모드를
모두 제공할 수 있을 때만 `true`로 보고합니다.
`mode_readiness.group`과 `mode_readiness.member`로 모드별 준비 상태를 따로 확인할 수
있습니다. 멤버 트래커는 일반 모드에는 선택 요소지만 멤버 모드에는
필수입니다. `member_anchor_tracking`은 CSRT/KCF/MIL 팩토리 존재만 보는 것이
아니라 작은 텍스처 프레임에서 생성·초기화·업데이트를 실제로 검증한
백엔드만 `mode_capabilities`에 품질 우선순위대로 표시합니다. 전역
`optional_features`에는 폴백이 있는 기능만 두며, 각 번들 모델의
정확한 SHA-256과 OpenCV 로드 가능 여부를 따로 보여 줍니다. `yt-dlp`는 진단 정보에만 표시되는
선택 도구이며 이 앱은 다운로드 기능에 사용하지 않습니다.

```bash
cd engine
python3 crosscut_cli.py doctor
python3 -m unittest discover -s tests -v
```

```bash
cd CrossCutMac
xcodegen generate
xcodebuild -project CrossCutMac.xcodeproj -scheme CrossCutMac test
```

앱이 엔진을 자동으로 찾지 못하면 `CROSSCUT_ENGINE` 환경 변수에 `engine/crosscut_cli.py`의 절대 경로를 지정할 수 있습니다.

## 검증 범위

- 실제 FFmpeg를 사용한 가로·세로 H.264/AAC 렌더 통합 테스트
- 분수 프레임 컷, 동적 팬 크롭, 밀집 키프레임 회귀 테스트
- 로컬 수동 통합 smoke에서 서로 다른 오디오 지연을 가진 9개 합성 직캠을
  모두 정렬·추적하고 9:16 실제 MP4 렌더
- 동일 조건의 9개 안전 후보를 완곡 타임라인에 고르게 분산하는 회귀 테스트
- 번들된 포즈·실루엣 모델의 SHA-256, OpenCV DNN 로드 및 실제 추론 검증
- 번들된 SFace의 SHA-256, `FaceRecognizerSF` 로드 및 128값 임베딩 검증
- 포즈 실패 시 구조 폴백, 군무 실루엣 연속성 선택, 실루엣 결측 중립 처리

합성 테스트는 파이프라인과 시간축 계약을 검증하기 위한 것입니다. 실제 IVE 영상의 조명, 가림, 카메라 흔들림에 대한 결과는 별도의 실영상 검증이 필요합니다.

## 알고리즘 출처와 라이선스

얼굴 전환의 핵심 아이디어는 Tobigs의 Apache-2.0 프로젝트
[`Cross-Cutting`](https://github.com/datamarket-tobigs/Cross-Cutting)이 사용한
눈 좌표와 주변 프레임 minimax 비교를 참고했습니다. 기존 dlib/MoviePy 코드를
복사하지 않고, 오디오 동기화된 YuNet 5점 랜드마크와 현재 전역 DP 구조에 맞춰
독립적으로 다시 구현했습니다. 세부 출처와 포함하지 않은 모델의 라이선스 경계는
[`docs/PROVENANCE.md`](docs/PROVENANCE.md)에 기록합니다.

신체 연속성 분석은 공식
[`opencv/opencv_zoo`](https://github.com/opencv/opencv_zoo)의 Apache-2.0
FP32 ONNX 모델 세 개(MediaPipe Person Detector, MediaPipe Pose Estimator,
PP-HumanSeg)를 번들해 로컬에서 실행합니다. 정확한 업스트림 커밋, 모델
SHA-256, 라이선스 사본은
[`engine/crosscut/models/PROVENANCE.pose.md`](engine/crosscut/models/PROVENANCE.pose.md)에
함께 기록했습니다. 영상 프레임은 네트워크로 전송하지 않습니다.

A 등급의 얼굴 외형 호환성 확인에는 같은 OpenCV Zoo의 Apache-2.0
[`SFace`](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface)
FP32 모델을 사용합니다. 정확한 SHA-256과 이 기능을 A 등급 후보에만 제한한
라이선스·개인정보 경계는
[`docs/PROVENANCE.md`](docs/PROVENANCE.md)에 기록했습니다.
