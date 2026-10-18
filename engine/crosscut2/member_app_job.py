"""Normal-app bridge for complete, source-bound cached MEMBER observations.

A cache stores measurements/alignment, never the chosen cut list. Every job
selects the whole programme again. A complete matching movie may then be reused
only after the new plan and its content-bound physical frame clock agree.
Cold observation acquisition and GROUP integration are separate work.
"""
from dataclasses import replace
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,time
import os
from pathlib import Path

import numpy as np

from crosscut.ive import EditMode,AspectRatio,SourceMode
from crosscut.render_plan_serialization import render_plan_from_json
from crosscut.render_reference_clock import load_render_reference_clock
from crosscut.render_clock_artifact import render_clock_path
from .member_stage_mix import Source
from .member_geometry_inputs import identity,load_geometry,augment_faces
from .member_body_context import ObservedBodyContext
from .member_programme import build
from .stage_mix import load_warps
from .visible_shot_duration import Model,exposure_tail_artifact

INPUT_CONTRACT='current_member_numeric_inputs_v1'
ARTIFACT_CONTRACT='current_member_programme_review_v1'


def input_path(request):
    if request.source_cache_directory is None or request.target_member is None:return None
    return request.source_cache_directory/'crosscut2'/f'member-inputs-{request.target_member.name.lower()}.json'


def _read(bound):
    raw=Path(bound['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=bound['sha256']:raise ValueError('cached member input dependency changed')
    return json.loads(raw)


def load_inputs(path,request):
    raw=Path(path).read_bytes();value=json.loads(raw)
    if value.get('contract')!=INPUT_CONTRACT or value.get('member')!=request.target_member.value:
        raise ValueError('current member cache has a different target/schema')
    if ''.join(value.get('song','').casefold().split())!=''.join(request.song.casefold().split()):return None
    entries=value['sources'];by_path={str(Path(e['path']).resolve()):i for i,e in enumerate(entries)}
    if len(by_path)!=len(entries):raise ValueError('duplicate physical sources in current member inputs')
    try:order=[by_path[str(p.resolve())] for p in request.sources]
    except KeyError:return None
    if len(set(order))!=len(order):raise ValueError('duplicate requested member source')
    if order[request.reference_source]!=value['reference_source']:
        return None  # another master audio needs its own independent alignment
    sources=[Source(Path(entries[i]['path']),entries[i]['fps'],entries[i]['width'],entries[i]['height'],entries[i]['frame_count']) for i in order]
    identities=[entries[i]['identity'] for i in order]
    if any(identity(s.path)!=bound for s,bound in zip(sources,identities,strict=True)):
        raise OSError('current member source bytes/stat identity changed')
    geometry=load_geometry(Path(value['geometry']['path']),namespace=value['geometry']['namespace'],sources=sources,identities=identities)
    remap={old:new for new,old in enumerate(order)}
    def supplements(records):
        result=[]
        for bound in records:
            doc=_read(bound)
            result.append(dict(doc,sources=[dict(row,source=remap[row['source']]) for row in doc['sources'] if row['source'] in remap]))
        return result
    faces=supplements(value['face_inputs'])
    if not faces:raise ValueError('current member cache needs its observed target-query inputs')
    spaces={(doc['gallery_identity'],json.dumps(doc['descriptor_space'],sort_keys=True)) for doc in faces}
    observed,merge=augment_faces(geometry,sources=sources,identities=identities,supplements=faces,member=request.target_member.value)
    queries=[{} for _ in sources]
    for doc in faces:
        for source in doc['sources']:
            for row in source['observations']:
                bank=queries[source['source']];old=bank.get(row['native'])
                if old is not None and old['raster_sha256']!=row['raster_sha256']:raise ValueError('current face query pixels disagree')
                bank.setdefault(row['native'],row)
    body=[]
    for new,i in enumerate(order):
        rows={}
        for row in value['body_by_source'][i]:
            frame,native=row['frame'],row['native'];people=row['people']
            face=queries[new].get(native)
            if face is None or face['raster_sha256']!=people['raster_sha256']:
                raise ValueError('body context and current face query do not bind the same native image')
            people=dict(people,keypoints17=np.asarray(people['keypoints17'],float).reshape(-1,17,3),scores=np.asarray(people['scores'],float))
            rows[frame]=(native,people)
        body.append(rows)
    positive=[{n:any(f['member']==request.target_member.value for f in row['faces']) for n,row in q.items()} for q in queries]
    context=ObservedBodyContext(body,positive,fps=value['fps'])
    for doc in supplements(value.get('overlap_face_inputs',[])):
        if (doc['gallery_identity'],json.dumps(doc['descriptor_space'],sort_keys=True)) not in spaces:
            raise ValueError('overlap face query uses a different descriptor/reference space')
        for source in doc['sources']:
            if source['source_file_identity']!=identities[source['source']]:raise OSError('overlap source binding changed')
            for row in source['observations']:
                bank=queries[source['source']];old=bank.get(row['native'])
                if old is not None and old['raster_sha256']!=row['raster_sha256']:raise ValueError('overlap query pixels disagree')
                bank.setdefault(row['native'],row)
    model=Model(exposure_tail_artifact(_read(value['duration_model']),fps=value['fps']))
    warps=load_warps({'local_timewarps':value['local_timewarps']})
    return dict(sources=sources,observations=observed,original_geometry=geometry,current_faces=queries,
        warps=[warps[i] for i in order],master_start=value['master_start'],duration_frames=value['duration_frames'],
        fps=value['fps'],beat_times=value['beat_times'],duration_model=model,body_context=context,
        source_order=order,face_merge=merge,cached_movies=value.get('completed_movies',[]),
        inputs_sha256=hashlib.sha256(raw).hexdigest())


def try_analyze(request,reporter):
    """Use current cached observations on an ordinary analyze call, or miss.

    Explicit learned-model/profile requests retain their existing path. An
    invalid present observation bundle is an error, not a silent old rebuild.
    No successful render is called professional/delivery quality here.
    """
    path=input_path(request)
    from crosscut.stage_edit_production_registry import STAGE_EDIT_PRODUCTION_REGISTRY_ENV
    if (request.edit_mode is not EditMode.MEMBER or request.aspect_ratio is not AspectRatio.PORTRAIT
            or path is None or not path.is_file() or len(request.sources)<2 or request.acquire_only
            or request.source_mode is not SourceMode.AUTOMATIC
            or STAGE_EDIT_PRODUCTION_REGISTRY_ENV in os.environ
            or request.render_range_start_seconds is not None or any(getattr(request,k,None) is not None for k in (
                'professional_reference_profile','professional_reference_cadence_profile',
                'stage_edit_checkpoint_directory','stage_edit_production_training_report',
                'stage_edit_generation_checkpoint_directory','stage_edit_generation_feature_snapshot',
                'stage_edit_learned_reviewer_checkpoint_directory','stage_edit_learned_reviewer_production_training_report'))):return None
    data=load_inputs(path,request)
    if data is None:return None
    started=time.monotonic();reporter.status('현재 멤버 편집기로 전체 곡을 구성하는 중…')
    request.output_directory.mkdir(parents=True,exist_ok=True)
    slug='IVE-'+request.song.replace('/','-')+'-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output=request.output_directory/(slug+'-member-crosscut.mp4')
    bound_identities=[identity(s.path) for s in data['sources']]
    def check():
        if any(identity(s.path)!=e for s,e in zip(data['sources'],bound_identities,strict=True)):
            raise OSError('requested member source order/binding changed')
    kwargs={k:data[k] for k in ('warps','master_start','duration_frames','fps','beat_times','duration_model','body_context','current_faces','original_geometry')}
    plan,programme=build(data['sources'],data['observations'],output=output,
        reference_source=request.reference_source,member=request.target_member.value,
        effects_enabled=request.member_pose_dissolve_enabled,check=check,**kwargs)
    reused=False
    for candidate in data['cached_movies']:
        movie=Path(candidate['path'])
        if not movie.is_file() or not render_clock_path(movie).is_file():continue
        raw=json.loads(render_clock_path(movie).read_bytes());expected=replace(plan,output=movie)
        if raw.get('plan')!=json.loads(json.dumps(expected.to_json())):continue
        with movie.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=candidate['sha256']:raise ValueError('complete cached movie content changed')
        load_render_reference_clock(movie,json.loads(json.dumps(expected.to_json())),())
        plan=expected;reused=True;break
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:raise OSError('current member app path needs FFmpeg')
    if reused:
        reporter.log('새 전체 편집안이 검증된 기존 출력과 동일해 영상 인코딩을 재사용합니다.')
        decoded=subprocess.run([ffmpeg,'-nostdin','-v','error','-threads','2','-i',str(plan.output),
            '-map','0:v','-map','0:a','-f','null','-'],capture_output=True,timeout=90.)
        if decoded.returncode or decoded.stderr:raise ValueError('cached complete member movie failed AV decode')
        rendering=dict(movie=str(plan.output),movie_sha256=sha,full_AV_decode_exit_code=0,source_NN=0,
            full_film_cache_reused=True,video_encoder_calls=0,frame_clock_artifact=str(render_clock_path(plan.output)))
    else:
        from .bounded_plan_render import render
        rendering=render(plan,ffmpeg=ffmpeg,maximum_seconds=660.,check=check,
            progress_callback=lambda e:reporter.emit('log',**dict(e,stage='current_member_render')))
    saved_plan=request.output_directory/(slug+'-render-plan.json')
    saved_plan.write_text(json.dumps(plan.to_json(),indent=2,allow_nan=False)+'\n')
    from crosscut.review_first_runtime import ReviewFirstQADisposition,ReviewFirstStatus
    disposition=ReviewFirstQADisposition(ReviewFirstStatus.REVIEW_READY,'review_ready',True,False,
        editorial_issue_codes=('normal_speed_editorial_review_pending',)).to_dict()
    manifest=dict(contract=ARTIFACT_CONTRACT,created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        request=request.as_dict(),artifact_role='review_ready',
        professional_quality_approved=False,backend='current_member_programme_v1',inputs=str(path),
        inputs_sha256=data['inputs_sha256'],source_order=data['source_order'],programme=programme,
        render=dict(rendering,video_path=str(plan.output),plan_path=str(saved_plan),artifact_role='review_ready'),
        app_result_exposure=dict(video_output_exposed=True,delivery_completion_message_allowed=False),
        post_render_qa=dict(review_first_disposition=disposition),
        elapsed_seconds=time.monotonic()-started,warnings=['검토용 출력입니다. 전문 편집 품질 판정은 완료되지 않았습니다.'])
    manifest_path=request.output_directory/(slug+'-analysis.json')
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    reporter.progress(1.,'현재 멤버 편집안과 검토용 출력을 준비했습니다.')
    from crosscut.pipeline import AnalysisArtifact
    return AnalysisArtifact(manifest_path,manifest,(manifest_path,saved_plan,plan.output))


def exposable_review(artifact):
    """Recognize this backend's real review film, never an approved delivery."""
    manifest=artifact.manifest
    if manifest.get('contract')!=ARTIFACT_CONTRACT:return False
    request=manifest.get('request',{})
    from crosscut.review_first_runtime import artifact_disposition_payload_valid
    if not artifact_disposition_payload_valid(manifest.get('post_render_qa',{}).get('review_first_disposition'),role='review_ready'):
        raise ValueError('current member output is not explicitly review-only')
    if (request.get('group')!='IVE' or request.get('edit_mode')!='member'
            or request.get('source_mode') not in ('local','automatic') or not request.get('target_member')):
        raise ValueError('current member review has no valid IVE request scope')
    videos=[Path(p) for p in artifact.output_paths if Path(p).suffix.lower()=='.mp4']
    rendering=manifest.get('render',{});exposure=manifest.get('app_result_exposure',{})
    if (len(videos)!=1 or not videos[0].is_absolute()
            or manifest.get('artifact_role')!='review_ready' or manifest.get('professional_quality_approved') is not False
            or rendering.get('artifact_role')!='review_ready' or rendering.get('full_AV_decode_exit_code')!=0
            or rendering.get('video_path')!=str(videos[0]) or exposure.get('video_output_exposed') is not True
            or exposure.get('delivery_completion_message_allowed') is not False):
        raise ValueError('current MEMBER result is not a content-bound review film')
    plan=json.loads(Path(rendering['plan_path']).read_bytes())
    if plan['output']!=str(videos[0]):raise ValueError('current member movie differs from its newly selected programme')
    if plan['sources']!=request.get('sources'):raise ValueError('current member review source domain changed')
    if plan.get('member_pose_dissolve_requested') is not request.get('member_pose_dissolve_enabled'):
        raise ValueError('current member effect preference changed at render publication')
    with videos[0].open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=rendering['movie_sha256']:raise ValueError('current member film changed after media-integrity verification')
    load_render_reference_clock(videos[0],plan,())
    return True
