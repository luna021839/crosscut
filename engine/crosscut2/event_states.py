"""Freeze measured shot cameras before selection; share source observations.

This opt-in domain does not use the legacy comfort-band unary or moving
one-frame camera fits. Its loss rates and event seconds are produced together
with event_ports. Anonymous actor paths remain hypotheses, not named members.
"""
from dataclasses import replace
import numpy as np
from .fixed_holds import make_states,_curve_at
from .shot_camera_paths import make_paths


def crop_added_loss(body,projected):
    valid=(body[...,2]>=.3)&(body[...,:2]>=0).all(-1)&(body[...,:2]<=1).all(-1)
    clipped=valid&((projected[...,:2]<0).any(-1)|(projected[...,:2]>1).any(-1))
    return clipped.sum(-1)/np.maximum(1,valid.sum(-1))


def displayed_attention(heads,camera,index):
    """Source-only visible face salience under an ACTUAL candidate crop.

    A selected background face cannot stand in for the larger visible focal
    person in the rendered pixels. This is an attention proxy, not ownership.
    It is computed before any cross-source match/member comparison.
    """
    center=.5+camera[0]*(heads[:,:2]-camera[1:]);height=heads[:,2]*camera[0]
    half=height[:,None]*[.45*9/16,.5]
    extent=np.maximum(0.,np.minimum(1.,center+half)-np.maximum(0.,center-half))
    area=np.prod(extent,axis=1)
    salience=area*np.exp(-((center[:,0]-.5)/.4)**2)
    salience=np.where((heads[:,2]>.019)&(heads[:,3]>=.75),salience,0.)
    maximum=float(salience.max())
    ratio=0. if index<0 or maximum<=0 else float(salience[index]/maximum)
    return ratio,bool(index>=0 and maximum>0 and salience[index]>=maximum*(1.-1e-6))


def build(nodes,banks,poses,styles,*,fps,source_events,body_states,check=lambda:None,identity_hints=None,part_observer=None):
    if len(styles)!=len(banks):raise ValueError('source-bound native style inputs required')
    bases,raw,counts=make_states(nodes,banks,poses,fps=fps,source_events=source_events,
        body_states=body_states,check=check,camera_families=('native',))
    part_receipts=[]
    if part_observer is not None:raw,part_receipts=part_observer(bases,raw)
    paths,camera_report=make_paths(nodes,banks,source_events,check=check)
    states=[];data=[]
    for state,row in zip(bases,raw):
        check();source=state.source;indices=row['indices'];native=banks[source]
        # Exact source observation for the same local actor index, not the
        # largest face in a different crop or a fallback to another person.
        quality=np.full(indices.shape,.5,np.float32);colour=np.zeros(indices.shape+(2,40),np.float32)
        colour_valid=np.zeros(indices.shape,bool)
        for n,side in zip(*np.nonzero(indices>=0)):
            h=int(indices[n,side]);style=styles[source]
            colour[n,side]=style['colour'][n,side,h];colour_valid[n,side]=style['colour_valid'][n,side,h]
            if native.get('native_readability_valid',np.zeros((0,),bool)).shape==native['heads'].shape[:3] and native['native_readability_valid'][n,side,h]:
                contrast,edge,clipped,height=native['native_readability'][n,side,h]
                quality[n,side]=min(1.,height/120.)*min(1.,edge/.5)*min(1.,contrast/.075)*(1.-clipped)
        salience=np.exp(-row['utility_parts']['salience']/1.1)
        for family,(ticks,curve) in enumerate(paths[source]):
            check();cameras=_curve_at(ticks,curve,np.clip(nodes[:,None]+[-1,0],0,nodes[-1]-1))
            four=_curve_at(ticks,curve,np.clip(nodes[:,None]+[-7,-1,0,6],0,nodes[-1]-1))
            projected=row['body'].copy()
            projected[...,:2]=.5+four[:,:,None,:1]*(projected[...,:2]-four[:,:,None,1:])
            crop=crop_added_loss(row['body'],projected).reshape(len(nodes),2,2).max(2)
            head=row['heads'];display=np.minimum(1.,head[:,:,2]*cameras[:,:,0]/.22)
            center=.5+cameras[:,:,:1]*(head[:,:,:2]-cameras[:,:,1:])
            half=head[:,:,2,None]*cameras[:,:,:1]*np.array([.45*9/16,.5])
            extent=np.maximum(0.,np.minimum(1.,center+half)-np.maximum(0.,center-half))
            face_fraction=np.prod(extent/np.maximum(1e-8,2.*half),axis=-1)
            # Enlargement changes displayed framing, never native pixel quality.
            attention=np.ones(indices.shape,np.float32);dominant=np.zeros(indices.shape,bool)
            member_hint=np.full(indices.shape,-1,np.int16)
            for n,side in zip(*np.nonzero(indices>=0)):
                attention[n,side],dominant[n,side]=displayed_attention(native['heads'][n,side],cameras[n,side],int(indices[n,side]))
                owner=int(row['owner'][n,side])
                if identity_hints is not None and owner>=0:
                    hints=identity_hints[source]
                    if owner>=len(hints):raise ValueError('source-local member hint lost its actor domain')
                    member_hint[n,side]=int(hints[owner])
            availability=np.sqrt(np.maximum(0.,quality))*display*np.sqrt(salience)*face_fraction*attention
            presence=styles[source]['presence'];black=np.clip((presence[:,:,0]-.8)/.2,0,1)
            availability=np.clip(availability*(1.-black),0,1)
            losses=1.-availability+crop+.1*np.log(cameras[:,:,0])/np.log(2.)
            unary=(losses[:-1,1]+losses[1:,0])/2.
            record=dict(row,cameras=cameras,ticks=ticks,curve=curve,projected=projected,
                native_quality=quality,availability=availability,colour=colour,colour_valid=colour_valid,
                displayed_attention_ratio=attention,displayed_attention_dominant=dominant,local_member_hint=member_hint,
                added_crop=crop,event_loss_rate=unary,
                camera_candidate_is_not_verified_actor_track=True)
            # Do not accidentally send legacy objective arrays to the new DP.
            record.pop('unary');record.pop('through')
            states.append(replace(state,family='native' if family==0 else f'shot_match_{family-1}'));data.append(record)
    return tuple(states),data,dict(source_local_appearance_clusters=counts,fixed_states=len(states),
        native_part_corroboration=part_receipts,
        raw_actor_states_prepared_once=len(bases),camera_domain=camera_report,
        native_pixel_quality_not_increased_by_zoom=True,stage_edit_transformer_used=False,
        professional_quality_approved=False)
