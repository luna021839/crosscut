import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from crosscut.render import RenderPlan,RenderSegment,AspectRatio


def plan_at(tmp_path):
    return RenderPlan((Path('one.mp4'),Path('two.mp4')),
        (RenderSegment(0,1.,1.,0.),RenderSegment(1,2.,1.,1.)),0,
        AspectRatio.LANDSCAPE,tmp_path/'output.mp4',width=320,height=180,fps=30.)


@pytest.mark.parametrize('field',['decoder_threads','filter_complex_threads'])
@pytest.mark.parametrize('value',[True,0,17,'2',float('nan')])
def test_render_thread_limits_are_finite_explicit_integers_before_io(tmp_path,field,value):
    module=importlib.import_module('crosscut.render')
    with pytest.raises(ValueError,match=field):module.render(plan_at(tmp_path),**{field:value})


@pytest.mark.parametrize('decode,filters',[(2,2),(4,1)])
def test_every_opened_input_has_a_bound_and_quality_and_graph_are_unchanged(tmp_path,monkeypatch,decode,filters):
    module=importlib.import_module('crosscut.render');process=importlib.import_module('crosscut.render_process')
    plan=plan_at(tmp_path);graph=module.build_filter_graph(plan);commands=[]
    def stop(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1,stderr='intentional command inspection stop')
    monkeypatch.setattr(process,'run_observed_render',stop)
    with pytest.raises(RuntimeError,match='intentional command inspection'):
        module.render(plan,video_codec='command_test',use_verified_frame_schedule=False,
            decoder_threads=decode,filter_complex_threads=filters)
    command=commands[0];inputs=[i for i,v in enumerate(command) if v=='-i']
    assert len(inputs)==2
    for i in inputs:assert command[i-2:i]==['-threads',str(decode)]
    assert command[command.index('-filter_complex_threads')+1]==str(filters)
    for key,value in (('-b:v','24M'),('-maxrate','36M'),('-bufsize','48M'),('-b:a','320k')):
        assert command[command.index(key)+1]==value
    assert module.build_filter_graph(plan)==graph and not plan.output.exists()
