from datetime import UTC, datetime, timedelta
import json
import subprocess
from unittest import TestCase
from unittest.mock import patch

from crosscut.acquisition import YTDLPClient, AcquisitionError, _PROFESSIONAL_DOWNLOAD_PROFILE
from crosscut.source_format_refresh import _fresh, refresh_group_cached_formats
from crosscut.source_format_revision import resolve_group_format_cache
from engine.tests.test_source_format_revision import SourceFormatRevisionTests


class FormatMetadataTests(TestCase):
    def client(self,payload,code=0):
        def runner(command,**kwargs):
            self.command,self.kwargs=command,kwargs
            return subprocess.CompletedProcess(command,code,json.dumps(payload),'')
        return YTDLPClient('/mock/yt-dlp',runner=runner)

    def test_actual_formats_not_title_or_download_preference(self):
        client=self.client({'id':'abcdefghijk','title':'4K','formats':[
            {'vcodec':'h264','width':1920,'height':1080},
            {'vcodec':'av1','width':3840,'height':2160},
            {'vcodec':'none','width':9999,'height':9999}]})
        self.assertEqual(client.best_video_dimensions('abcdefghijk'),(3840,2160))
        self.assertIn('--no-cookies',self.command)
        self.assertIn('--skip-download',self.command)
        self.assertLessEqual(self.kwargs['timeout'],40)

    def test_false_4k_title_does_not_invent_4k_format(self):
        self.assertEqual(self.client({'id':'abcdefghijk','title':'4K','formats':[
            {'vcodec':'h264','width':1920,'height':1080}]}).best_video_dimensions('abcdefghijk'),(1920,1080))

    def test_wrong_id_empty_formats_and_failed_process_rejected(self):
        for payload,code in [({'id':'wrong','formats':[]},0),({'id':'abcdefghijk','formats':[]},0),
                             ({'id':'abcdefghijk','formats':[]},1)]:
            with self.assertRaises(AcquisitionError):self.client(payload,code).best_video_dimensions('abcdefghijk')

    def test_availability_cache_binds_content_profile_and_age(self):
        now=datetime.now(UTC)
        record={'content_integrity':{'sha256':'a'},'profile':'p','checked_at':now.isoformat(),
            'decision':'best_available_dimensions_already_cached'}
        self.assertTrue(_fresh(record,binding={'sha256':'a'},profile='p',now=now))
        self.assertFalse(_fresh(record,binding={'sha256':'b'},profile='p',now=now))
        self.assertFalse(_fresh(record,binding={'sha256':'a'},profile='other',now=now))
        self.assertFalse(_fresh(record,binding={'sha256':'a'},profile='p',now=now+timedelta(days=8)))


class CachedFormatRefreshTests(SourceFormatRevisionTests):
    def refresh(self,client):
        return refresh_group_cached_formats(request=self.request,artifact=self.artifact,
            reporter=self.reporter,client=client,media_validator=self.probe,
            media_probe=self.probe,group_quality_analyzer=None)

    def test_success_prepares_validates_and_activates_source_revision(self):
        outer=self
        class Client:
            download_profile=_PROFESSIONAL_DOWNLOAD_PROFILE
            def best_video_dimensions(self,video_id):
                return (3840,2160) if video_id==outer.video_id else (1920,1080)
            def download(self,candidate,directory,*,timeout_seconds):
                outer.assertLessEqual(timeout_seconds,600)
                return outer.new
        result=self.refresh(Client())
        self.assertNotEqual(result.manifest_path,self.artifact.manifest_path)
        self.assertEqual(resolve_group_format_cache(self.request),result.manifest_path.parent)

    def test_network_failure_retains_every_original_and_is_not_retried_immediately(self):
        class Client:
            download_profile=_PROFESSIONAL_DOWNLOAD_PROFILE
            calls=0
            def best_video_dimensions(self,video_id):
                self.calls+=1
                raise AcquisitionError('offline')
        client=Client()
        self.assertIs(self.refresh(client),self.artifact)
        count=client.calls
        self.assertIs(self.refresh(client),self.artifact)
        self.assertEqual(client.calls,count)
        self.assertEqual(count,8)
        self.assertTrue(all(p.is_file() for p in self.artifact.source_paths))

    def test_revision_validation_failure_preserves_original_active_cache(self):
        outer=self
        class Client:
            download_profile=_PROFESSIONAL_DOWNLOAD_PROFILE
            def best_video_dimensions(self,video_id):return (3840,2160)
            def download(self,candidate,directory,*,timeout_seconds):return outer.new
        with patch('crosscut.source_format_refresh.prepare_group_format_revision',side_effect=ValueError('invalid')):
            self.assertIs(self.refresh(Client()),self.artifact)
        self.assertEqual(resolve_group_format_cache(self.request),self.request.source_cache_directory)
