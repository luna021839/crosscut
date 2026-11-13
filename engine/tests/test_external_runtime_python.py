import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import external_runtime_python as runtime


UUID='12345678-1234-5678-9abc-123456789abc'


def configuration(tmp_path):
    image=tmp_path/'runtime.sparseimage';image.write_bytes(b'fixture image')
    path=tmp_path/'local.json'
    data=dict(schema=1,image_path=str(image),mount_path='/Volumes/CrossCut-Test',
        python_relative='python-3.11/bin/python3',volume_uuid=UUID)
    path.write_text(json.dumps(data));return path,data


def test_existing_volume_is_verified_without_installing_or_reattaching(tmp_path,monkeypatch):
    path,_=configuration(tmp_path)
    monkeypatch.setattr(runtime,'mounted_uuid',lambda p:UUID)
    monkeypatch.setattr(Path,'is_file',lambda p:True)
    monkeypatch.setattr(runtime.os,'access',lambda *a:True)
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:pytest.fail('no subprocess for an existing verified mount'))
    python,attached=runtime.resolve_python(path)
    assert python==Path('/Volumes/CrossCut-Test/python-3.11/bin/python3') and not attached


def test_wrong_volume_cannot_run_or_replace_another_drive(tmp_path,monkeypatch):
    path,_=configuration(tmp_path)
    monkeypatch.setattr(runtime,'mounted_uuid',lambda p:'different-volume')
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**kw:pytest.fail('must not mount over another volume'))
    with pytest.raises(ValueError,match='UUID differs'):runtime.resolve_python(path)


def test_cold_mount_attaches_once_then_checks_the_actual_volume_uuid(tmp_path,monkeypatch):
    path,_=configuration(tmp_path);states=iter((None,UUID));calls=[]
    monkeypatch.setattr(runtime,'mounted_uuid',lambda p:next(states))
    monkeypatch.setattr(Path,'is_mount',lambda p:False)
    original=Path.exists;monkeypatch.setattr(Path,'exists',lambda p:False if p==Path('/Volumes/CrossCut-Test') else original(p))
    monkeypatch.setattr(runtime.os,'access',lambda *a:True)
    real_file=Path.is_file
    monkeypatch.setattr(Path,'is_file',lambda p:True if p.name=='python3' else real_file(p))
    monkeypatch.setattr(runtime.subprocess,'run',lambda args,**kw:(calls.append(args) or SimpleNamespace(returncode=0)))
    python,attached=runtime.resolve_python(path)
    assert attached and len(calls)==1 and calls[0][0]=='/usr/bin/hdiutil'
    assert str(python).startswith('/Volumes/CrossCut-Test/')


@pytest.mark.parametrize('field,value',[
    ('python_relative','../../other/python'),('python_relative','/usr/bin/python3'),
    ('mount_path','/'),('mount_path','/Users/someone'),('volume_uuid','not-a-uuid'),('schema',True)])
def test_invalid_local_configuration_is_rejected(tmp_path,field,value):
    path,data=configuration(tmp_path);data[field]=value;path.write_text(json.dumps(data))
    with pytest.raises(ValueError):runtime.load_configuration(path)


def test_missing_image_cannot_fall_back_to_internal_python(tmp_path):
    path,data=configuration(tmp_path);Path(data['image_path']).unlink()
    with pytest.raises(ValueError):runtime.load_configuration(path)
