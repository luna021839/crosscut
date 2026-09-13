"""One audited equivalence for previously COMPLETE bridge scalars only.

Observed-face recovery runs only after the old optical-flow function returned
None. The old provider returned evidence_complete=False immediately on that
path, and incomplete dispositions have never been admitted to scalar shards.
Consequently every formerly persistable disposition follows unchanged logic.
This is not a general algorithm migration and transfers no renderer proofs.
"""
from copy import deepcopy
from collections.abc import Mapping

from .joint_scalar_reader_compatibility import MEASURED_FACE_ANALYZER, NATIVE_TRACKING_READER_PROVIDER

RECOVERY_PROVIDER = '18e9fd4d710577eb7a8a7f5ed81e9db7a5d3ad50e53a52d3c7e7dc46ebdb881a'
RECOVERY_CONTAINMENT = '8dd96e54c327d1befd42c7ead2e2f475a5d3d6a0272bfbbd480a941e1ad78c21'
RECOVERY_IMPLEMENTATION = '5dde17d3be9f60830aa602e9d462784be2443848cc044c7aacab01860e3f466e'
PREVIOUS_CONTAINMENT = '8f54d7e725c6ecdcf48129cf1833147677f68dd959e421d0e8a6cab161611a11'


def previous_complete_containment_analyzer(analyzer):
    if analyzer.get('analysis_contract') != 'joint-bridge-disposition-scalar-shard-v3':
        return None
    versions=analyzer.get('visual_analyzer_sha256s')
    sources=analyzer.get('source_content_sha256s')
    implementation=analyzer.get('implementation_sha256')
    if (not isinstance(versions,list) or not isinstance(sources,list)
            or len(versions)<2 or len(versions)!=len(sources)
            or any(v!=MEASURED_FACE_ANALYZER for v in versions)
            or not isinstance(implementation,Mapping)):
        return None
    expected={'provider_class_ast':RECOVERY_PROVIDER,
              'cut_anchored_containment_sha256':RECOVERY_CONTAINMENT,
              'observed_face_containment_sha256':RECOVERY_IMPLEMENTATION}
    if any(implementation.get(k)!=v for k,v in expected.items()):
        return None
    previous=deepcopy(dict(analyzer))
    old=previous['implementation_sha256']
    old['provider_class_ast']=NATIVE_TRACKING_READER_PROVIDER
    old['cut_anchored_containment_sha256']=PREVIOUS_CONTAINMENT
    del old['observed_face_containment_sha256']
    return previous
