# Architecture

## 앱과 엔진의 경계

`CrossCutMac/Sources/Models/EditConfiguration.swift`가 작업 입력을 구성하고, `EngineRunner.swift`가 Python 프로세스를 실행합니다. 엔진은 진행 이벤트와 결과 경로를 전달하며, 앱은 작업 상태와 결과 노출을 관리합니다. 취소 처리는 엔진과 하위 미디어 프로세스의 수명 주기를 함께 다룹니다.

Python 진입점은 `engine/crosscut_cli.py`입니다. `crosscut` 패키지는 수집·동기화·영상 처리와 공통 계약을, `crosscut2`는 관측 기반 편집 선택과 개인 영상 효과를 제공합니다. 기존 파이프라인과 새 편집 모듈이 함께 남아 있어, 작은 단일 엔진으로의 통합은 아직 진행 중입니다.

## 음악 시간축과 영상 시간축

여러 공연은 시작 시간이 다르고 중간의 재생 속도도 조금씩 다릅니다. 오디오 정렬 결과로 소스별 시간을 공통 음악 시간축에 대응시킨 뒤, 후보 관측과 최종 렌더 모두 같은 시간 관계를 사용합니다.

관련 코드:

- `crosscut/audio.py`: 오디오 특징과 정렬
- `crosscut2/reference_clock.py`, `programme_source_clock.py`: 편집·소스 시간 대응
- `crosscut2/source_cuts.py`, `native_timing.py`: 원본 카메라 컷과 전환 시점

## 컷의 연결성

컷 선택은 두 정지 프레임의 얼굴 위치만으로 결정하지 않습니다. 양쪽 구간의 얼굴 크기, 포즈, 팔과 몸의 배치, 이동 방향과 화면 내 피사체 위치를 비교합니다. 카메라 변환 뒤 실제로 보이는 영역도 평가 대상입니다.

관련 코드:

- `crosscut2/focal_output_correspondence.py`, `paired_geometry.py`: 화면상 대응 관계
- `crosscut2/subject_action_match.py`, `motion_direction.py`: 동작 연결
- `crosscut2/boundary_readability.py`, `native_readability.py`: 컷 양쪽의 가독성
- `crosscut2/shot_camera_paths.py`, `observed_reframing.py`: 카메라 경로

## 전체 타임라인

국소적으로 점수가 높은 컷만 반복하면 샷이 지나치게 짧아지거나 같은 구도가 이어질 수 있습니다. 후보 전환을 음악 시간축 위의 경로로 구성하고, 연결 비용·샷 유지 시간·문맥을 함께 평가합니다. 동적 계획법과 제한된 탐색 경로는 필요한 관측과 작업량에 따라 구분되어 있습니다.

관련 코드:

- `crosscut2/boundary_port_dp.py`, `opportunity_dp.py`: 후보 경로 선택
- `crosscut2/played_pose_port_programme.py`: 포즈와 카메라를 결합한 프로그램 구성
- `crosscut2/sequence_context.py`, `visible_shot_duration.py`: 문맥과 샷 길이
- `crosscut2/playback_workspace.py`, `bounded_suffix_spool.py`: 탐색 작업 공간

## 개인 직캠과 의상 전환

개인 영상은 동작과 신체 구도를 이어 붙이는 편집과, 의상 영역의 변화를 강조하는 후처리를 나눕니다. 디졸브가 얼굴 전체의 색 변화로 보이지 않도록 얼굴 색감과 의상 전환 영역을 별도로 다룹니다.

관련 코드:

- `crosscut2/member_programme.py`, `member_pair_camera.py`: 개인 영상 편집과 카메라
- `crosscut2/member_short_dissolve.py`: 짧은 전환
- `crosscut2/member_wardrobe_layer.py`, `member_wardrobe_finish.py`: 의상 교체 효과
- `crosscut2/member_face_tone.py`: 얼굴 색감

## 캐시와 모델

관측 캐시를 이용해 같은 소스를 반복 디코딩하는 비용을 줄입니다. 소스·모델·관측 방식이 달라졌을 때 이전 값을 잘못 재사용하지 않도록 바인딩을 검사합니다. 원본 관측과 선택된 카메라·타임라인을 구분하여, 편집 선택을 바꿀 때 필요한 계산만 다시 수행하는 구조입니다.

기본 검출 모델과 추가 학습 가중치는 별개입니다. 얼굴 가중치나 아티스트 카탈로그의 구체적인 값이 편집 알고리즘 자체를 정의하지 않습니다. 일부 연구용 실행 도구에는 외장 볼륨 검사와 추가 모델 경로가 필요하므로, 해당 도구의 인자와 실행 환경을 먼저 지정해야 합니다.
