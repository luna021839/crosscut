import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

SPEC=importlib.util.spec_from_file_location('current_engine_bundler',Path(__file__).parents[2]/'tools/bundle_engine.py')
bundler=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(bundler)


def inputs(tmp_path):
    source=tmp_path/'source';build=tmp_path/'build';build.mkdir()
    destination=build/'Example.app/Contents/Resources/engine'
    destination.mkdir(parents=True);(destination/'old.py').write_text('old intact')
    for name in (*bundler.REQUIRED,'crosscut2/member_programme.py'):
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'print(1)\n' if path.suffix=='.py' else b'asset')
    return source,destination,build


def test_validated_bundle_replaces_only_the_built_engine(tmp_path):
    source,destination,build=inputs(tmp_path);result=bundler.bundle(source,destination,build)
    assert not (destination/'old.py').exists()
    assert (destination/'crosscut2/member_programme.py').read_bytes()==(source/'crosscut2/member_programme.py').read_bytes()
    assert 'crosscut2/member_programme.py' in result['files']


def test_corrupted_copy_never_replaces_the_previous_engine(tmp_path):
    source,destination,build=inputs(tmp_path)
    def corrupt(a,b):Path(b).write_bytes(b'\0corrupt')
    with patch.object(bundler.shutil,'copyfile',side_effect=corrupt),pytest.raises(ValueError,match='copy differs'):
        bundler.bundle(source,destination,build)
    assert (destination/'old.py').read_text()=='old intact'


def test_workspace_root_cannot_be_a_cleaned_destination(tmp_path):
    source,destination,build=inputs(tmp_path)
    with pytest.raises(ValueError,match='outside this built app'):bundler.bundle(source,tmp_path,build)
    assert source.exists() and (destination/'old.py').exists()
