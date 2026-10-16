import pytest
from crosscut.group_source_quality import _camera_variant,_group_structure
from crosscut.acquisition import _is_capture_variant,_ordered_for_download,_metadata_candidate
from test_acquisition import _entry,_request


@pytest.mark.parametrize('marker',['#음중팔로우캠4K','팔로우 캠','FollowCam','Follow Cam','Follow-Cam'])
def test_follow_camera_is_auxiliary_not_a_canonical_programme_feed(tmp_path,marker):
    request=_request(tmp_path)
    main=_metadata_candidate(_entry(0,title='IVE - I AM | Music Bank Stage 230410'),
        request=request,query='IVE I AM')
    entry=_entry(0,title=f'[{marker}] IVE - I AM | Music Bank Stage 230410')
    entry['id']='follow00000';entry['webpage_url']='https://www.youtube.com/watch?v=follow00000'
    following=_metadata_candidate(entry,request=request,query='IVE I AM')
    assert main is not None and following is not None
    assert _camera_variant(following.title)=='follow_cam'
    assert _group_structure(following.title)==(.94,('group_followcam_structure',))
    assert _is_capture_variant(following) and not _is_capture_variant(main)
    assert _ordered_for_download([following,main])==[main,following]


def test_real_mbc_followcam_title_is_not_mistaken_for_standard_programme():
    title='[#음중팔로우캠4K] IVE (아이브) – XOXZ | 쇼! 음악중심 | MBC250830 | KOR/ENG/JPN/ESP Lyrics'
    assert _group_structure(title)[1]==('group_followcam_structure',)
    assert _group_structure('IVE (아이브) - XOXZ | Show! MusicCore | MBC250830방송')[1]==(
        'standard_music_show_programme_feed',)
