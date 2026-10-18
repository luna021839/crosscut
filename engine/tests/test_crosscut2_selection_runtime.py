import pytest
from crosscut2.stage_mix import main,selection_runtime_limit


def test_paired_interval_deadline_covers_measured_solve_without_changing_legacy_modes():
    assert selection_runtime_limit(paired_cameras=True,played_intervals=True)==900.
    assert selection_runtime_limit(paired_cameras=True,played_intervals=False)==300.
    assert selection_runtime_limit(paired_cameras=False,played_intervals=True)==180.
    assert selection_runtime_limit(paired_cameras=False,played_intervals=False)==180.
    assert selection_runtime_limit(paired_cameras=True,played_intervals=True,explicit_seconds=840.)==840.


@pytest.mark.parametrize('seconds',[True,0.,-1.,1201.,float('nan'),float('inf')])
def test_deadline_is_finite_and_bounded(seconds):
    with pytest.raises(ValueError,match='finite selection allowance'):
        selection_runtime_limit(paired_cameras=True,played_intervals=True,explicit_seconds=seconds)


@pytest.mark.parametrize('seconds',['nan','0','1201'])
def test_cli_rejects_bad_deadline_before_reading_media(monkeypatch,tmp_path,seconds):
    argv=['stage_mix','--maximum-selection-seconds',seconds]
    for key in ('source-plan','analysis','cache','output','models'):argv.extend(['--'+key,str(tmp_path/key)])
    monkeypatch.setattr('sys.argv',argv)
    with pytest.raises(ValueError,match='finite selection allowance'):main()


def test_cli_rejects_unused_allowance_before_source_work(monkeypatch,tmp_path):
    argv=['stage_mix','--maximum-selection-seconds','840']
    for key in ('source-plan','analysis','cache','output','models'):argv.extend(['--'+key,str(tmp_path/key)])
    monkeypatch.setattr('sys.argv',argv)
    with pytest.raises(ValueError,match='requires the joint motion-event'):main()
