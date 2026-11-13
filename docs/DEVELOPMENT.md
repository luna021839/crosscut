# 개발 과정

2026-08-23부터 2026-10-18까지의 개발 기록을 정리했습니다. 각 단계의 링크에서 당시 코드 변경과 테스트를 확인할 수 있습니다.

- [개발 커밋 전체 보기](https://github.com/luna021839/crosscut/commits/codex/development-history/)
- [현재 공개본의 커밋 로그](https://github.com/luna021839/crosscut/commits/codex/portfolio/)
- [대표 시연](../README.md#대표-시연)

## 주요 변경

| 시점 | 변경 내용 | 실제 변경 기록 |
| --- | --- | --- |
| 2026-08-23 | macOS 앱과 영상 수집·동기화·교차편집의 초기 작업 흐름 구성 | [초기 구현](https://github.com/luna021839/crosscut/commit/1ab2175e6189495ba424a4295b229d7e95f8a87f) |
| 2026-08-24 | 소스별 카메라 차이를 다루는 학습 기반 크롭과 후보 평가 실험 | [소스 조건부 크롭](https://github.com/luna021839/crosscut/commit/5c4c5c544358a5470ce572784c04e43ccd7d5f6a) |
| 2026-10-09 | 출력 영상의 기하 평가와 작은 전체 곡 편집 모듈을 분리해 구성 | [출력 기하 평가](https://github.com/luna021839/crosscut/commit/b1caa28adb9252195ecc10c309024968dec71aaa), [편집 모듈 구성](https://github.com/luna021839/crosscut/commit/beb1a117ee5654cef56c24efbfd10ccf46a9efcd) |
| 2026-10-14~16 | 컷 양쪽의 동작, 유지 구간의 카메라 움직임, 인물 대응과 카메라 후보를 함께 평가 | [동작·카메라 연속성](https://github.com/luna021839/crosscut/commit/51fc7e080050855c90a8cb5a11cd1d34e288d07b), [인물 대응 후 카메라 선택](https://github.com/luna021839/crosscut/commit/c10df6a06dcfb8ccf363ae353f110d26d8343478) |
| 2026-10-18 | 개인 직캠의 세로 화면 비율을 반영한 관측·카메라 검증과 앱 연결 | [세로 뷰포트](https://github.com/luna021839/crosscut/commit/7931ee8b0e19e8f6da5a3ee3375f8be702de632a), [개인 영상 효과의 앱 연결](https://github.com/luna021839/crosscut/commit/bfca61684e3f7949149327281cf2760c81b8e688) |
| 2026-10-18 | 의상·얼굴·무대의 전환을 분리하고 얼굴 색감 보정 추가 | [영역별 전환](https://github.com/luna021839/crosscut/commit/7a1f29cadaeaec7d047310f533219bc6d60291a9), [얼굴 색감 보정](https://github.com/luna021839/crosscut/commit/d422890952e6a7a0995edc4b0074aa4108621fea) |
| 2026-11-13 | 포트폴리오 문서, 선별 시연과 공개 가능한 개발 이력 정리 | [공개본 준비](https://github.com/luna021839/crosscut/commit/e55cab12400879a25d16dee49af651fa6358df76), [담당 범위·결과 정리](https://github.com/luna021839/crosscut/commit/6da38ebd5d94703645402d9df5b13f9cfe00be44) |

## 개선 과정에서 바꾼 판단

### 음악 정렬 이후에도 남는 동작 단절

오디오가 맞아도 컷 양쪽의 얼굴 크기, 팔의 배치와 카메라 움직임이 다르면 연결이 어색했습니다. 컷 시점의 얼굴만 비교하는 데서 나아가, 같은 음악 시간의 동작과 실제로 재생되는 구간의 화면을 평가하도록 수정했습니다. 인물의 대응 관계와 카메라 후보를 타임라인 선택 단계에서 함께 다루도록 구조를 옮겼습니다.

[포즈와 카메라 후보를 선택 전에 구성한 변경](https://github.com/luna021839/crosscut/commit/60796f89e1e93406708bb3f13de5669e9ec4bc57)에서 이 흐름을 확인할 수 있습니다. README의 I AM과 XOXZ 발췌는 결과 영상에서 선별한 연결 사례입니다.

### 편집 수정 때 반복되는 분석 비용

컷이나 카메라를 바꿀 때 원본 관측까지 다시 계산하면 비교 결과를 받는 시간이 길어졌습니다. 원본에서 얻은 관측과 선택된 카메라·타임라인의 평가를 나누고, 소스 시간축과 모델 정보가 일치하는 관측을 재사용하도록 개선했습니다. 효과만 바뀐 경우에는 이미 인코딩된 장면을 활용하는 경로도 추가했습니다.

재사용 범위는 소스·모델·시간 정보에 연결해 관리합니다. 관련 구현은 [관측 캐시 계약](../engine/crosscut2/observation_cache_contract.py)과 [인코딩 장면 재사용](../engine/crosscut2/encoded_programme_reuse.py)에서 볼 수 있습니다.

### 의상 전환이 얼굴 색 변화로 보이는 문제

화면 전체를 한 번에 섞으면 의상이 바뀌는 표현이 약해지고 얼굴과 배경의 색 차이가 함께 드러났습니다. 의상 영역의 전환을 얼굴·무대 전환과 분리하고, 전환 구간의 얼굴 색감을 이어주는 처리를 추가했습니다. README에는 개선한 쪽만 발췌한 6초 시연을 넣었습니다.

관련 코드는 [의상 전환 레이어](../engine/crosscut2/member_wardrobe_layer.py)와 [얼굴 색감 보정](../engine/crosscut2/member_face_tone.py)에 있습니다.

## 공개 이력의 구성

원본 개발 저장소의 758개 커밋 중 공개 가능한 파일의 변경이 남는 **721개 커밋**을 옮겼습니다. 비공개 문서·검토 도구·운영 설정 등 제외 대상만 바뀐 37개 커밋은 공개 이력에서 빠졌습니다.

작성·커밋 시각과 코드 변경 순서를 유지하고, 작성자 표기를 현재 포트폴리오 계정으로 맞췄습니다. 개인 경로와 비공개 자료를 정리하면서 커밋 ID는 변경되었습니다. 공개 대상 파일의 과거 버전 4,341개와 커밋 메타데이터에서 알려진 개인 식별자, 비공개 대화 주소, 인증정보 형태를 검사했습니다.

기존 공개 스냅샷의 커밋 위에 개발 이력을 병합해, 현재 공개 파일과 이전 개발 과정을 함께 볼 수 있도록 구성했습니다. 개발 단계의 실험 코드와 현재 사용 경로에 대한 설명은 [아키텍처 문서](ARCHITECTURE.md)에 정리했습니다.
