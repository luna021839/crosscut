"""Bounded fair numeric conditional inventory; never an observation receipt.

The ordinary global API used a lexicographic Cartesian prefix, which could
exclude every alternative of its first unit. Rank-sum ordering visits each
unit's first alternative before higher-order combinations. A retained cursor
owns the exact inventory and unexecuted combinations; restoring this NUMERIC
cursor does not restart, grant or resume a native observation/work budget.
"""
from hashlib import sha256
from heapq import heapify, heappop, heappush
import json
from math import prod

CONTRACT='bounded_global_scene_conditional_frontier_v1'
MAX_CONDITIONAL_PATHS=32
MAX_SCOPED_UNITS=18


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


class ConditionalSceneFrontier:
    def __init__(self,groups,pairs,*,maximum_paths=MAX_CONDITIONAL_PATHS):
        if (not isinstance(groups,tuple) or not 1<=len(groups)<=MAX_SCOPED_UNITS
                or any(not isinstance(g,tuple) or len(g)>32 for g in groups)
                or type(maximum_paths) is not int or not 1<=maximum_paths<=MAX_CONDITIONAL_PATHS
                or not isinstance(pairs,tuple) or tuple(sorted(set(pairs)))!=pairs
                or any(not isinstance(p,tuple) or len(p)!=2 or
                    any(type(i) is not int for i in p) or not 0<=p[0]<p[1]<len(groups)
                    or p[1]!=p[0]+1 for p in pairs)):
            raise ValueError('conditional frontier needs a bounded canonical unit/pair inventory')
        self.groups=groups;self.pairs=pairs;self.maximum_paths=maximum_paths
        self.keys=tuple(tuple(s.key for s in group) for group in groups)
        if any(len(set(keys))!=len(keys) or any(not isinstance(k,str) or not k for k in keys)
                for keys in self.keys):
            raise ValueError('conditional frontier keys must be distinct within every original unit')
        self.inventory=dict(ordered_state_keys=[list(keys) for keys in self.keys],
            incident_pair_indices=[list(pair) for pair in pairs],maximum_paths=maximum_paths)
        self.inventory_sha256=sha256(_json(self.inventory).encode()).hexdigest()
        self.total=prod(map(len,groups))
        origin=(0,)*len(groups)
        self.heap=[(0,origin)] if self.total else []
        self.seen={origin} if self.total else set()
        self.completed=[];self.active=None;self.infeasible={}

    def peek(self):
        if self.active is not None:
            return tuple(self.groups[i][v] for i,v in enumerate(self.active))
        if not self.heap or len(self.completed)>=self.maximum_paths:return None
        vector=self.heap[0][1]
        return tuple(self.groups[i][v] for i,v in enumerate(vector))

    def begin(self):
        if self.active is not None or self.peek() is None:
            raise ValueError('conditional frontier has no unstarted live candidate')
        _,self.active=heappop(self.heap)
        for i in range(len(self.groups)):
            if self.active[i]+1>=len(self.groups[i]):continue
            child=list(self.active);child[i]+=1;child=tuple(child)
            if child not in self.seen:
                self.seen.add(child);heappush(self.heap,(sum(child),child))
        return self.peek()

    def finish(self,status):
        if self.active is None or not isinstance(status,str) or not status:
            raise ValueError('conditional frontier needs one explicit completed outcome')
        self.completed.append(dict(rank_vector=list(self.active),status=status,
            state_keys=[self.keys[i][v] for i,v in enumerate(self.active)]))
        self.active=None

    def remember_infeasible_pair(self,pair,left,right,reason):
        if (pair not in self.pairs or left.key not in self.keys[pair[0]]
                or right.key not in self.keys[pair[1]] or not isinstance(reason,str) or not reason):
            raise ValueError('conditional pair rejection differs from its frozen unit inventory')
        self.infeasible[(*pair,left.key,right.key)]=reason

    def blocked_pair(self,values):
        for i,j in self.pairs:
            key=(i,j,values[i].key,values[j].key)
            if key in self.infeasible:
                return dict(pair_indices=[i,j],state_keys=[values[i].key,values[j].key],
                    reason=self.infeasible[key],previous_observed_pair_reused=True)
        return None

    @property
    def exhaustive(self):return self.total>0 and len(self.completed)==self.total

    @property
    def outcome(self):
        return ('HYPOTHESES_UNAVAILABLE' if not self.total else
            'COMPLETE_NO_GAIN' if self.exhaustive else 'SEARCH_TRUNCATED')

    def to_dict(self):
        return dict(contract=CONTRACT,inventory=self.inventory,inventory_sha256=self.inventory_sha256,
            total_combinations=self.total,completed_combination_count=len(self.completed),
            pending_combination_count=self.total-len(self.completed),completed=self.completed,
            active_rank_vector=None if self.active is None else list(self.active),
            next_rank_vector=(list(self.active) if self.active is not None else
                list(self.heap[0][1]) if self.heap else None),
            queued_rank_vectors=[list(v) for _,v in sorted(self.heap)],
            discovered_rank_vectors=[list(v) for v in sorted(self.seen)],
            infeasible_pairs=[dict(pair_indices=list(k[:2]),state_keys=list(k[2:]),reason=v)
                for k,v in sorted(self.infeasible.items())],
            ordering='rank_sum_then_vector_not_lexicographic_cartesian_prefix',
            inventory_exhausted=self.exhaustive,unexecuted_combinations_are_unknown=True,
            numeric_cursor_is_not_native_resume_authority=True,new_native_budget_granted=False,
            quality_approved=False)

    @classmethod
    def from_dict(cls,groups,pairs,cursor):
        """Validate a numeric continuation without inventing any native work."""
        try:
            result=cls(groups,pairs,maximum_paths=cursor['inventory']['maximum_paths'])
            for row in cursor['completed']:
                result.begin();result.finish(row['status'])
            if cursor['active_rank_vector'] is not None:result.begin()
            for row in cursor['infeasible_pairs']:
                i,j=row['pair_indices'];a,b=row['state_keys']
                result.remember_infeasible_pair((i,j),
                    next(s for s in groups[i] if s.key==a),next(s for s in groups[j] if s.key==b),row['reason'])
            if _json(result.to_dict())!=_json(cursor):
                raise ValueError('conditional cursor changed its complete canonical progress')
            heapify(result.heap)
            return result
        except (KeyError,TypeError,IndexError,StopIteration) as exc:
            raise ValueError('conditional cursor lost its frozen numeric inventory') from exc
