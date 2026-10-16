"""Independent finite enumeration; synthetic cues are NOT fitted AV labels."""
from contextlib import nullcontext
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.compiled_context_port_dp import solve as compiled_solve
from crosscut2.context_port_dp import solve as reference_solve,ArcRef
from crosscut2.compiled_played_context import _Origin
from crosscut2.playback_input_batch import pack
from crosscut2.playback_suffix_arena import PlaybackSuffixArena

NODES=np.array([0,12,24,37]);H=12
ENDS=np.r_[np.arange(6,37,6),37];DURATIONS=np.diff(np.r_[0,ENDS])/30.
TEMPORAL=np.column_stack((np.arange(H)*.001,np.sin(np.arange(H))*.003))


def sample(ref):
    ticks=np.arange(NODES[ref.first],NODES[ref.final]);family=ref.camera%2;source=ref.entry
    native=ticks+100*source
    camera=np.tile([1.+.1*family,.5,.5],(len(ticks),1))
    _,keys=pack(model_binding=('numeric-only','never-a-fitted-artifact'),frames=ticks,native=native,cameras=camera,
        source_observation_handles=np.full(len(ticks),source,np.int64),intent_handles=np.zeros((len(ticks),2),np.int64))
    encoded=np.column_stack((np.full(len(ticks),source*.04+family*.01),.0001*(ticks%11)))
    return ticks,keys,encoded


def temporal_cost(ticks,encoded,ends):
    costs=[]
    for end in ends:
        value=.05
        for t,v in zip(ticks,encoded):
            if max(0,end-H)<=t<end:
                for d in range(2):value+=v[d]*TEMPORAL[end-1-t,d]
        costs.append(np.logaddexp(0.,value)*DURATIONS[np.searchsorted(ENDS,end)])
    return float(np.asarray(costs).sum())


def reference_extra(histories,ref):
    result=[];fresh=ENDS[(ENDS>NODES[ref.first])&(ENDS<=NODES[ref.final])]
    for history in histories:
        records=[sample(r) for r in (*history,ref)]
        ticks=np.concatenate([r[0] for r in records]);values=np.concatenate([r[2] for r in records])
        result.append(temporal_cost(ticks,values,fresh))
    return np.asarray(result)


def physical_key(history,end):
    rows=np.concatenate([sample(r)[1] for r in history]) if history else np.zeros((0,8),np.uint64)
    return tuple(tuple(map(int,r)) for r in rows[rows[:,0]>=max(0,end-H)])


class Current:
    lower_bound=0.
    def __init__(self,compiler,ref):
        self.ref=ref;self.frames,self.keys,self.encoded=sample(ref)
        self.lease=compiler.arena.admit(self.keys[-H:],self.encoded[-H:],first=int(NODES[ref.first]))
    def __enter__(self):return self
    def __exit__(self,*unused):self.lease.close()


class Hold:
    def __init__(self,compiler,i,j):self.compiler=compiler;self.i=i;self.j=j
    def current(self,p,q,code):return Current(self.compiler,ArcRef(self.i,self.j,p,q,code))
    def __enter__(self):return self
    def __exit__(self,*unused):return False


class Compiler:
    def __init__(self,kernel):
        self.arena=PlaybackSuffixArena(namespace=('numeric-independent-enumeration-only',),embedding_width=2,
            history_frames=H,frame_capacity=512,cell_capacity=128,root_capacity=128,kernel=kernel)
        self.context=SimpleNamespace(nodes=NODES,model=SimpleNamespace(embedding_width=2));self.origin_decode_calls=0
        self.active_origin=None
    def open_origin(self,i,roots,costs):return _Origin(self,i,roots,costs)
    def reserve_hold(self,i,j):return nullcontext(SimpleNamespace(observe=lambda *a:None))
    def compile_hold(self,i,j,base,allowed):return Hold(self,i,j)
    def score(self,bank,current,p,bit,slots):
        fresh=ENDS[(ENDS>NODES[current.ref.first])&(ENDS<=NODES[current.ref.final])];result=[]
        for slot in slots:
            lane=bank.lanes[p,bit,slot]
            ticks=np.r_[bank.decoded.frames,current.frames]
            encoded=np.concatenate((bank.decoded.values[lane],current.encoded))
            result.append(temporal_cost(ticks,encoded,fresh))
        return np.asarray(result)


def base_cost(i,j,code,source):return .1+source*.02+code*np.finfo(float).eps


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_six_code_aliases_paid_and_unpaid_states_terminal_window_match_complete_enumeration(kernel):
    if kernel=='numba':pytest.importorskip('numba')
    sources=[np.arange(2)]*len(NODES)
    def hold(i,j):
        base=np.zeros((2,2,6));allowed=np.zeros(base.shape,bool)
        for s in range(2):
            for code in range(6):base[s,s,code]=base_cost(i,j,code,s);allowed[s,s,code]=j==i+1
        return base,allowed
    kw=dict(start_cost=np.zeros(2),start_allowed=[True,False],final_allowed=[True,False],
        hold_cost=hold,cut_cost=lambda n:(np.zeros((2,2)),~np.eye(2,dtype=bool)),
        episodes=np.array([0,0,0,1]),credit=lambda i,j:np.full(2,.125 if i else 0.),
        history_frames=H,label_limit=8)
    old=reference_solve(NODES,sources,sources,context_cost=reference_extra,history_key=physical_key,**kw)
    compiler=Compiler(kernel);new=compiled_solve(NODES,sources,sources,compiler=compiler,**kw)
    assert new[0]==old[0] and new[1]==old[1]
    assert new[2]['selected_arc_codes']==old[2]['selected_arc_codes']
    assert new[2]['opportunity_claims']==old[2]['opportunity_claims']
    # All 6^3 camera-code programmes and every legal once-only paid/no-paid
    # placement, independently accumulated in the original FP operation order.
    exhaustive=[]
    for codes in product(range(6),repeat=3):
        refs=[ArcRef(i,i+1,i%2,i%2,code) for i,code in enumerate(codes)]
        extra=[reference_extra([tuple(refs[:i])],ref)[0] for i,ref in enumerate(refs)]
        for claims in ((False,False,False),(False,True,False),(False,False,True)):
            total=0.
            for i,(code,sequence,paid) in enumerate(zip(codes,extra,claims)):
                total=((total+base_cost(i,i+1,code,i%2))+sequence)-(.125 if paid else 0.)
            exhaustive.append(total)
    assert new[1][0]==min(exhaustive)
    assert sum(c['claimed'] for c in new[2]['opportunity_claims'])==1
    assert new[2]['camera_alternatives_scored']==36
    assert all(t['live']==0 for t in compiler.arena.receipt()['tables'].values())
    compiler.arena.close()
