"""Fixed music episodes and physically supported same-action exposure.

Episode boundaries precede selection and use observed pairs of beats, not a
desired shot length. The paid bit belongs to an episode, not a source. Missing
body/track evidence removes this credit only; it never forbids a cut or hold.
"""
import numpy as np


class Opportunities:
    def __init__(self,nodes,music,*,fps,families,beat_frames=None):
        self.nodes=np.asarray(nodes,int);self.music=np.asarray(music,float);self.fps=float(fps)
        if self.music.shape!=self.nodes.shape or not np.isfinite(self.music).all() or np.any((self.music<0)|(self.music>1)):
            raise ValueError('fixed measured music strengths required')
        self.families=tuple(families)
        self.continuous_strength=beat_frames is not None
        beats=self.nodes[self.music>0] if beat_frames is None else np.asarray(beat_frames)
        if beat_frames is not None and not beats.size:beats=np.zeros(0,np.int64)
        if beat_frames is not None and (beats.ndim!=1 or beats.dtype.kind not in 'iu' or np.any(np.diff(beats)<=0)
                or np.any((beats<=0)|(beats>=self.nodes[-1]))):
            raise ValueError('distinct observed interior beat frames required')
        self.boundaries=np.unique(np.r_[0,beats[::2],self.nodes[-1]])
        self.episodes=np.searchsorted(self.boundaries,self.nodes,side='right')-1
        self.utility=np.zeros(max(0,len(self.boundaries)-1));self.bindings={};self.exposures={}

    def observe(self,n,left,right,detail,left_sources,right_sources):
        """All physical raw pairs BEFORE global8 camera proposal pruning."""
        for a,la in enumerate(left):
            for b,rb in enumerate(right):
                sa,sb=int(left_sources[a]),int(right_sources[b])
                if self.families[sa]==self.families[sb]:continue
                value=float(detail['matched_action_change_density'][a,b]*self.music[n])
                if value<=0:continue
                key=(n,sa,sb,int(la['head_index']),int(rb['head_index']))
                self.bindings[key]=max(value,self.bindings.get(key,0.))
                e=int(self.episodes[n]);self.utility[e]=max(self.utility[e],value)

    def density(self,n,left,right,left_source,right_source,actual):
        key=(n,int(left_source),int(right_source),int(left['head_index']),int(right['head_index']))
        # Changed cameras must still satisfy the real connection. A raw
        # retrieval opportunity is not proof of the played pose/viewport.
        return min(float(actual*self.music[n]),self.bindings.get(key,0.))

    def exposure(self,n,observation):
        """Only observed incoming frames; stop at gaps, cuts, or target loss."""
        context=observation.get('dense_subject_context')
        if context is None:return 0.  # no invented dense support from two sparse samples
        window=context['window'];at=context['at'];ticks=context['clock'].output_frames
        camera=np.asarray(observation['camera_strip'],float);used=0
        e=int(self.episodes[n]);end=int(self.boundaries[e+1])
        for tick in range(min(7,end-int(self.nodes[n]))):
            positions=np.flatnonzero(ticks==tick)
            if len(positions)!=1:break
            k=int(positions[0]);w=int(at[k])
            # An actually seen complete arm supports the target action;
            # shoulder-only observations cannot extend the useful exposure.
            arm=window.support[w]
            if (not window.associated[w] or not context['same_shot'][k]
                    or not (arm[[0,2,4]].all() or arm[[1,3,5]].all())):break
            if tick and not context['contiguous'][k-1]:break
            from .subject_action_match import _visibility
            visible,attention=_visibility(window.heads[w],camera[tick],window.competitors[w])
            if min(visible,attention)<.6:break
            used+=1
        return used/self.fps

    def credit(self,n,density,exposure):
        """Fixed event utility times coverage of its observed action strip.

        Utility has the same event-second units as the two measured beats.
        Coverage is of the seven ACTUALLY observed incoming frames, not a
        claim that the target persists throughout those musical seconds.
        Paying density*strip_seconds alone confuses a point action opportunity
        with a long content reward and removes almost every useful change.
        """
        e=int(self.episodes[n]);seconds=(self.boundaries[e+1]-self.boundaries[e])/self.fps
        utility=self.utility[e]*seconds
        ratio=np.minimum(1.,np.asarray(density)/max(1e-12,self.utility[e]))
        coverage=np.minimum(1.,np.asarray(exposure)/(7./self.fps))
        return utility*ratio*coverage

    def receipt(self):
        return dict(contract='fixed_nonoverlapping_observed_music_opportunity_v1',
            event_kind='MATCH_ACTION_CHANGE',episode_boundaries=self.boundaries.tolist(),
            music_strength_continuous_not_extra_beat_events=self.continuous_strength,
            episode_utility_density=self.utility.tolist(),raw_source_actor_bindings=len(self.bindings),
            credit_contract='fixed_event_utility_times_actual_seven_frame_action_coverage',
            full_hold_target_persistence_claimed=False,
            source_families=len(set(self.families)),family_duplicates_add_no_utility=True,
            current_episode_claimed_at_most_once=True,credit_never_extends_unobserved_target=True,
            no_minimum_cut_count_or_maximum_hold=True,role_handoff_objective_implemented=False,
            formation_reveal_objective_implemented=False,professional_quality_approved=False)
