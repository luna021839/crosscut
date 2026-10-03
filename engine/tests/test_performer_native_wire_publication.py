"""Real transaction publication of explicit NONHUMAN typed partial evidence.

The expensive proposer/context are substitutes. Publication and its byte limit
are production code; no media, model, parent/CAS/render or editorial pass occurs.
"""
from dataclasses import dataclass
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from crosscut import group_performer_family_owner as family
from crosscut.group_scene_native_storage import NativeObservationView,pack_native_observation,CONTRACT
from engine.tests.test_scene_region_flow import world


@dataclass(frozen=True)
class ExplicitNonhumanToken:
    permission: str='NO_PARENT_CAS_RENDER_OR_QUALITY_AUTHORITY'


def fixture():
    _,rows,_=world();raw=rows[0].to_dict()
    old=SimpleNamespace(to_dict=lambda:dict(explicit_nonhuman_predecessor=True),
        render_plan=SimpleNamespace(to_json=lambda:dict(explicit_nonhuman_plan=True)))
    controller=SimpleNamespace(incumbent=old,run_optional=lambda name,work:work())
    context=SimpleNamespace(token=ExplicitNonhumanToken())
    return raw,controller,context


def attempt(controller,path):
    return family.attempt_performer_family(controller=controller,request=None,
        eligible_indices=(),offsets=(),timewarps=(),scene_cut_times=(),sync_scores=(),archive_path=path)


def test_actual_transaction_publishes_typed_partial_with_read_through_native_rows(tmp_path):
    raw,controller,context=fixture();path=tmp_path/'explicit-partial.json'
    view=NativeObservationView(pack_native_observation(raw))
    evidence=dict(outcome='BUDGET_PARTIAL',unexecuted_measurements_are_unknown=True,
        raw_prefix_completeness_certified=False,quality_approved=False,
        rows=[dict(native_observation=view,measurement_available=False)])
    rejection=family.PerformerFamilyRejection('EXPLICIT_NONHUMAN_TYPED_PARTIAL',evidence)
    with patch.object(family,'freeze_context',return_value=context),\
            patch.object(family,'propose_performer_family',return_value=rejection),\
            pytest.raises(ValueError,match='EXPLICIT_NONHUMAN_TYPED_PARTIAL'):
        attempt(controller,path)
    saved=json.loads(path.read_text())
    assert saved['status']=='rejected_incumbent_preserved'
    assert saved['error_type']=='ValueError'
    assert saved['producer_rejection_evidence']['outcome']=='BUDGET_PARTIAL'
    assert saved['producer_rejection_evidence']['rows'][0]['native_observation']['contract']==CONTRACT
    assert family.plain(saved)['producer_rejection_evidence']['rows'][0]['native_observation']==json.loads(json.dumps(raw))
    assert saved['quality_approved'] is False
    assert saved['producer_rejection_evidence']['raw_prefix_completeness_certified'] is False
    assert controller.incumbent.to_dict()==saved['original_bundle']


def test_publication_does_not_stringify_an_unsupported_opaque_return(tmp_path):
    _,controller,context=fixture();path=tmp_path/'explicit-opaque.json'
    rejection=family.PerformerFamilyRejection('EXPLICIT_UNSUPPORTED_RETURN',dict(opaque=object()))
    with patch.object(family,'freeze_context',return_value=context),\
            patch.object(family,'propose_performer_family',return_value=rejection),pytest.raises(TypeError):
        attempt(controller,path)
    saved=json.loads(path.read_text())
    assert saved['status']=='preparing' and saved['quality_approved'] is False
    assert 'producer_rejection_evidence' not in saved


def test_publication_preserves_the_archive_cap_instead_of_dropping_partial_rows(tmp_path):
    raw,controller,context=fixture();path=tmp_path/'explicit-cap.json'
    rejection=family.PerformerFamilyRejection('EXPLICIT_TYPED_PARTIAL',dict(
        outcome='BUDGET_PARTIAL',rows=[dict(native_observation=NativeObservationView(pack_native_observation(raw)))]))
    with patch.object(family,'freeze_context',return_value=context),\
            patch.object(family,'propose_performer_family',return_value=rejection),\
            patch.object(family,'MAX_ARCHIVE_BYTES',512),\
            pytest.raises(ValueError,match='archive exceeds explicit byte budget'):
        attempt(controller,path)
    saved=json.loads(path.read_text())
    assert saved['status']=='preparing' and 'producer_rejection_evidence' not in saved
    assert saved['quality_approved'] is False
