"""Actual played action benefit of a PAIR-BOUND edit, not source content.

The existing episode paid bit prevents re-entry farming. Each incoming port
owns its original subject window; a different face or a better source-local
target cannot lend it evidence. Missing observations earn no benefit but do
not ban a source. Camera choice happens AFTER this actual interval benefit.
"""
import numpy as np
from .played_interval_utility import DeliverySamples,delivery_amount
from .boundary_arc_costs import cameras,camera_samples_many


class MatchedEntryDelivery:
    def __init__(self,arcs,opportunities):
        self.arcs=arcs;self.events=opportunities;self.nodes=arcs.nodes
        self.field=arcs.played_interval_utility;self.fps=arcs.fps
        if self.field is None or self.field.delivery_in_base:
            raise ValueError('matched-entry benefit requires the motion-only base, not a source-content reward')
        self._nodes={};self.calls=0;self.direct_camera_rows=0;self.observed_rows=0

    def _prepare(self,n):
        if n in self._nodes:return self._nodes[n]
        result=[];first=int(self.nodes[n]);episode=int(self.events.episodes[n])
        end=int(self.events.boundaries[episode+1])
        for port in self.arcs.entries[n]:
            context=port.observation.get('dense_subject_context')
            # Paired IDs are unique per physical outgoing/incoming actor
            # proposal. Static/native ports never acquire a change bonus.
            ids=[p for p in port.proposal_ids if p[0]=='paired']
            if context is None or not ids:
                result.append(None);continue
            if len(ids)!=1:raise ValueError('a claimed entry must bind one actual outgoing pair')
            source=port.source;window=context['window'];clock=self.field.clocks[source]
            frames=np.arange(first,end);times=np.interp(frames,self.nodes,clock['source_times'])
            native=(((times*1000.)/1000.)*clock['fps']+.5).astype(np.int64)
            at=native-int(window.ordinals[0]);inside=(at>=0)&(at<len(window.ordinals))
            frames,native,at=(v[inside] for v in (frames,native,at))
            endpoint=int(context['at'][np.flatnonzero(context['clock'].output_frames==0)[0]])
            shot=np.searchsorted(window.native_cuts,window.ordinals[endpoint],side='right')
            known=window.associated[at]&(np.searchsorted(window.native_cuts,native,side='right')==shot)
            frames,native,at=(v[known] for v in (frames,native,at))
            if not len(frames):result.append(None);continue
            table=DeliverySamples(frames,native,window.heads[at],window.body[at],window.support[at],
                tuple(window.competitors[k] for k in at),np.ones(len(frames)))
            # Long baseline arcs have exactly the same first19-frame
            # connector independently of their final node/exit port. Store
            # only observed rows and a tiny prefix per incoming port.
            paths=np.asarray(self.arcs.dense[source])[:,frames].copy()
            near=frames<first+19
            if near.any() and first+38<=int(self.nodes[-1]):
                for code in range(4):
                    baseline=self.arcs.dense[source][code][first:first+38]
                    curve,valid=cameras(port.strip[None],baseline[-7:][None],baseline,
                        cap=self.arcs.cats[source].bank['zoom_cap'])
                    if valid[0]:paths[code,near]=curve[0,frames[near]-first]
            amount=delivery_amount(table,np.arange(len(frames)),paths)/self.fps
            prefix=np.pad(np.cumsum(amount,axis=1),((0,0),(1,0)))
            result.append((table,amount,prefix));self.observed_rows+=len(frames)
        self._nodes[n]=result
        return result

    def benefits(self,i,j,density,allowed,*,direct=False):
        """Actual credit for every offered (entry, exit, camera) alternative.

        Long0..3 arcs use the same exact fixed interior/first19 prefix and
        correct only their final19 and any mismatching native ordinals.
        Short/whole-view families use their actual camera samples directly.
        No hold duration is used to extrapolate an unobserved body sample.
        """
        result=np.zeros(allowed.shape);self.calls+=1
        if not i:return result
        density=np.asarray(density,float)
        if (density.shape!=(len(self.arcs.entries[i]),) or not np.isfinite(density).all()
                or np.any(density<0)):
            raise ValueError('finite pair-bound incoming densities required')
        first,final=map(int,self.nodes[[i,j]]);tables=self._prepare(i)
        for p,item in enumerate(tables):
            if item is None or density[p]<=0:continue
            port=self.arcs.entries[i][p];source=port.source;table,amount,prefix=item
            used=int(np.searchsorted(table.frames,final));frames=table.frames[:used]
            if not used:continue
            native=self.field.native(source,first,final,frames);known=native==table.native[:used]
            for code in range(6):
                qs=np.flatnonzero(allowed[p,:,code])
                if not len(qs):continue
                compiled=not direct and final-first>=38 and code<4
                if compiled:
                    result[p,qs,code]=prefix[code,used]
                    positions=np.flatnonzero((frames>=final-19)|~known)
                    if not len(positions):continue
                else:positions=np.arange(used)
                sample_frames=frames[positions]
                baseline=self.arcs.dense[source][0 if code>=4 else code][first:final]
                entry=np.broadcast_to(port.strip,(len(qs),7,3))
                leave=np.asarray([self.arcs.exits[j][q].strip for q in qs])
                paths=camera_samples_many(entry,leave,baseline,sample_frames-first,code=code)
                actual=(delivery_amount(table,positions,paths)*known[positions]).sum(-1)/self.fps
                if compiled:actual-=amount[code,positions].sum()
                result[p,qs,code]+=actual
                self.direct_camera_rows+=len(qs)*len(positions)
        # Cancellation of a compiled prefix can leave roundoff below zero;
        # this is not manufactured exposure and never changes a positive row.
        return np.maximum(0.,result)*density[:,None,None]

    def choose(self,i,j,density,cost,allowed):
        benefit=self.benefits(i,j,density,allowed)
        eligible=allowed&(benefit>0)
        code=np.argmin(np.where(eligible,cost-benefit,np.inf),axis=-1)
        valid=eligible.any(-1)
        pick=code[...,None]
        return (np.take_along_axis(cost-benefit,pick,axis=-1).squeeze(-1),valid,code,
            np.take_along_axis(benefit,pick,axis=-1).squeeze(-1))

    def replay(self,i,j,p,curve,density):
        """Independent raw-row replay of the selected camera, not an argmin."""
        if not i:return 0.
        item=self._prepare(i)[p]
        if item is None:return 0.
        table=item[0];first,final=map(int,self.nodes[[i,j]])
        positions=np.flatnonzero(table.frames<final)
        if not len(positions):return 0.
        frames=table.frames[positions];source=self.arcs.entries[i][p].source
        native=self.field.native(source,first,final,frames)
        return float((delivery_amount(table,positions,np.asarray(curve)[None,frames-first])
            *(native==table.native[positions])).sum()/self.fps*density)

    def receipt(self):
        return dict(contract='pair_bound_actual_incoming_action_interval_once_per_event_v1',
            prepared_entry_nodes=len(self._nodes),observed_entry_target_rows=self.observed_rows,
            candidate_calls=self.calls,direct_camera_rows=self.direct_camera_rows,
            target_is_actual_entry_subject=True,benefit_before_camera_argmin=True,
            source_continuation_earns_no_change_credit=True,credit_only_in_entry_episode=True,
            no_seven_frame_extrapolation=True,no_hold_duration_extrapolation=True,
            no_new_source_analysis=True,no_cut_quota_or_maximum_hold=True,
            professional_quality_approved=False)
