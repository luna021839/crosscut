from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch
import errno

from crosscut.acquisition import acquire_sources, _content_integrity_matches
from crosscut.source_format_revision import (
    prepare_group_format_revision, activate_group_format_revision, resolve_group_format_cache,
)
from engine.tests.test_acquisition import _request, _entry, _FakeClient, _MemoryReporter, _landscape_media


class SourceFormatRevisionTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.request = _request(self.root, desired=8)
        self.reporter = _MemoryReporter()
        self.artifact = acquire_sources(self.request, self.reporter,
            client=_FakeClient([_entry(i) for i in range(8)]),
            media_validator=_landscape_media)
        self.video_id = self.artifact.manifest['selected'][0]['video_id']
        self.new = self.root/'staged'/f'{self.video_id}.mp4'
        self.new.parent.mkdir()
        self.new.write_bytes(b'new-format-fixture')

    def probe(self, path):
        info = _landscape_media(path)
        # Fixtures identify shared revision content through its bytes, not a
        # hardcoded revision path, so validation is repeated after linking.
        if path.read_bytes() == b'new-format-fixture':
            return replace(info, video_streams=(replace(info.video_streams[0],width=3840,height=2160),))
        return info

    def prepare(self, **kwargs):
        return prepare_group_format_revision(request=self.request,
            artifact=kwargs.pop('artifact',self.artifact),
            replacements=kwargs.pop('replacements',{self.video_id:self.new}),
            reporter=self.reporter, media_validator=self.probe,
            media_probe=kwargs.pop('media_probe',self.probe), **kwargs)

    def test_prepare_preserves_original_and_does_not_activate(self):
        old_manifest = self.artifact.manifest_path.read_bytes()
        revision = self.prepare()
        self.assertEqual(self.artifact.manifest_path.read_bytes(),old_manifest)
        self.assertEqual(resolve_group_format_cache(self.request),self.request.source_cache_directory)
        self.assertEqual(len(revision.source_paths),8)
        for entry in revision.manifest['selected']:
            self.assertTrue(_content_integrity_matches(Path(entry['local_path']),entry['content_integrity']))
        self.assertEqual(self.artifact.source_paths[0].read_bytes(),b'')
        self.assertFalse(revision.manifest['format_revision']['quality_approval'])

    def test_activation_changes_pointer_only(self):
        revision = self.prepare()
        activate_group_format_revision(request=self.request,revision_artifact=revision)
        self.assertEqual(resolve_group_format_cache(self.request),revision.manifest_path.parent)
        self.assertTrue(all(path.is_file() for path in self.artifact.source_paths))

    def test_normal_acquisition_consumes_activated_revision(self):
        revision=self.prepare()
        activate_group_format_revision(request=self.request,revision_artifact=revision)
        with patch('crosscut.acquisition._cache_refresh_upgrade_decision',return_value={'refresh_required':False}):
            actual=acquire_sources(self.request,self.reporter,client=_FakeClient([]),
                media_validator=self.probe,media_probe=self.probe)
        self.assertEqual(actual.source_paths,revision.source_paths)
        self.assertEqual(actual.manifest_path,revision.manifest_path)

    def test_mutation_between_prepare_and_activation_is_rejected(self):
        revision=self.prepare()
        revision.source_paths[0].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'content changed'):
            activate_group_format_revision(request=self.request,revision_artifact=revision)
        self.assertEqual(resolve_group_format_cache(self.request),self.request.source_cache_directory)

    def test_exfat_falls_back_to_exclusive_copy(self):
        with patch('crosscut.source_format_revision.os.link',side_effect=OSError(errno.ENOTSUP,'unsupported')):
            revision=self.prepare()
        self.assertTrue(all(path.is_file() for path in revision.source_paths))
        self.assertNotEqual(revision.source_paths[0].stat().st_ino,self.new.stat().st_ino)

    def test_changed_revision_content_falls_back_to_original_cache(self):
        revision=self.prepare()
        activate_group_format_revision(request=self.request,revision_artifact=revision)
        revision.source_paths[0].write_bytes(b'altered')
        self.assertEqual(resolve_group_format_cache(self.request),self.request.source_cache_directory)

    def test_another_upgrade_stays_in_original_root_not_nested_revisions(self):
        revision=self.prepare()
        other_id=revision.manifest['selected'][1]['video_id']
        other=self.new.parent/f'{other_id}.mp4'
        other.write_bytes(b'new-format-fixture')
        next_revision=self.prepare(artifact=revision,replacements={other_id:other})
        self.assertEqual(next_revision.manifest_path.parent.parent,self.request.source_cache_directory/'format-revisions')

    def test_unknown_video_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            self.prepare(replacements={'unknown1234':self.new})

    def test_same_resolution_codec_change_is_not_called_an_upgrade(self):
        with self.assertRaisesRegex(ValueError,'dimensions'):
            self.prepare(media_probe=_landscape_media)

    def test_changed_base_content_cannot_be_rebound(self):
        self.artifact.source_paths[0].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'base source content'):
            self.prepare()

    def test_unverified_excluded_pool_is_not_silently_lost(self):
        manifest=deepcopy(self.artifact.manifest)
        manifest['summary']['materialized_source_count']=9
        artifact=replace(self.artifact,manifest=manifest)
        with self.assertRaisesRegex(ValueError,'excluded'):
            self.prepare(artifact=artifact)

    def test_duration_mismatch_rejects_wrong_download(self):
        def changed_probe(path):
            info=self.probe(path)
            return replace(info,duration_seconds=190.) if path==self.new else info
        with self.assertRaisesRegex(ValueError,'duration'):
            self.prepare(media_probe=changed_probe)

    def test_symlink_replacement_is_rejected(self):
        linked=self.root/'linked'/self.new.name
        linked.parent.mkdir()
        linked.symlink_to(self.new)
        with self.assertRaisesRegex(ValueError,'regular'):
            self.prepare(replacements={self.video_id:linked})

    def test_pointer_traversal_and_wrong_song_fall_back(self):
        pointer=self.request.source_cache_directory/'active-format-revision.json'
        for revision,song in [('../elsewhere','I AM'),('a'*32,'another song')]:
            pointer.write_text(json.dumps({'contract':'group_source_format_revision_v1',
                'revision':revision,'song':song}))
            self.assertEqual(resolve_group_format_cache(self.request),self.request.source_cache_directory)
