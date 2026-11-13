"""Encoded-face diagnostics preserve the movie's actual physical geometry."""
import json
from types import SimpleNamespace
import pytest
import audit_rendered_focal_faces as audit


def probe(monkeypatch,stream):
    def run(command,**kwargs):
        assert command[-1]=='encoded.mp4' and kwargs['timeout']==10.
        return SimpleNamespace(stdout=json.dumps(dict(streams=[stream])))
    monkeypatch.setattr(audit.subprocess,'run',run)


@pytest.mark.parametrize('width,height',[(1280,720),(1920,1080),(3840,2160)])
def test_actual_native_group_raster_is_not_resized(monkeypatch,width,height):
    probe(monkeypatch,dict(width=width,height=height,sample_aspect_ratio='1:1'))
    assert audit.output_raster('encoded.mp4')==(width,height)


@pytest.mark.parametrize('change',[
    dict(width=1080,height=1920),dict(width=1440,height=1080),
    dict(width=7680,height=4320),dict(sample_aspect_ratio='4:3'),
    dict(sample_aspect_ratio='N/A'),dict(side_data_list=[dict(rotation=90)]),
])
def test_portrait_or_nonphysical_group_metric_is_rejected_before_face_inference(monkeypatch,change):
    probe(monkeypatch,dict(width=1920,height=1080,sample_aspect_ratio='1:1',**{})|change)
    with pytest.raises(ValueError,match='native square-pixel unrotated16:9 GROUP'):
        audit.output_raster('encoded.mp4')
