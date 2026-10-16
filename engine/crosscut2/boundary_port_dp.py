'''Exact lexicographic DP over separate entry/exit boundary ports.

Canonical node i owns ENTRY ports, where a hold may start (node 0 or after a
cut), and EXIT ports, where a hold may end (before a cut or at the final node).
Port domains may differ per node, and every port carries its source id.

hold_cost(i, j) -> (cost, allowed), shape (entries_i, exits_j).  The caller has
already folded physical content, played-native, fragment and the exact frozen
camera-path cost of that (entry, exit) pair into cost.  Cross-source pairs are
also masked here.
cut_cost(j) -> (cost, allowed), shape (exits_j, entries_j).  Same-source pairs
are also masked here.

Supplied costs must be finite everywhere.  Invalid arcs use the boolean mask,
never infinity.  Only internal DP states may be inf, meaning unreachable.
The objective is lexicographic: (total primary seconds, cut count).  Remaining
ties prefer the earliest start node and then the lowest port index.  Out states
are indexed by exit port, so no predecessor is cached independently of the exit
it serves.  Memory is O(N*P) values and backpointers plus one callback matrix at
a time.  Arcs are never stored.
'''
import numpy as np

__all__ = ['solve']
_FEWEST = np.iinfo(np.int64).max
_ARC_BYTES = 32  # cost, mask, admitted, candidate, tie and cut temporaries


def _ports(values, count, name):
    if len(values) != count:
        raise ValueError(f'{name} ports required for every canonical node')
    ports = []
    for value in values:
        a = np.asarray(value)
        if a.size == 0:
            a = np.zeros(0, np.int64)
        if a.ndim != 1 or a.dtype.kind not in 'iu' or np.any(a < 0):
            raise ValueError(f'{name} ports need 1-D non-negative integer source ids')
        ports.append(a.astype(np.int64))
    return ports


def _vector(value, size, name):
    a = np.asarray(value)
    if a.shape != (size,) or a.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a real vector of length {size}')
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite; use a mask for invalid ports')
    return a


def _mask(value, size, name):
    if value is None:
        return np.ones(size, bool)
    a = np.asarray(value)
    if a.shape != (size,) or a.dtype != np.bool_:
        raise ValueError(f'{name} must be a boolean vector of length {size}')
    return a


def _matrix(result, shape, name):
    try:
        cost, allowed = result
    except (TypeError, ValueError):
        raise ValueError(f'{name} callback must return (cost, allowed)') from None
    cost, allowed = np.asarray(cost), np.asarray(allowed)
    if cost.shape != shape or allowed.shape != shape:
        raise ValueError(f'{name} callback shapes {cost.shape}/{allowed.shape} != {shape}')
    if cost.dtype.kind not in 'iuf' or allowed.dtype != np.bool_:
        raise ValueError(f'{name} callback needs real costs and a boolean mask')
    cost = cost.astype(np.float64)
    if not np.isfinite(cost).all():
        raise ValueError(f'{name} costs must be finite; mask invalid arcs instead')
    return cost, allowed


def _lexmin(primary, cuts):
    '''Column-wise argmin by (primary, cuts, row index).'''
    low = primary.min(axis=0)
    tie = primary == low
    counted = np.where(tie, cuts, _FEWEST)
    fewest = counted.min(axis=0)
    return np.argmax(tie & (counted == fewest), axis=0), low, fewest


def solve(nodes, entry_sources, exit_sources, *, start_cost, hold_cost, cut_cost,
          start_allowed=None, final_cost=None, final_allowed=None, minimum_frames=12,
          check=lambda: None, workspace_limit_bytes=200 * 2**20,
          opportunity_episodes=None, hold_credit=None, claim_hold_cost=None):
    '''Return (runs, (primary, cuts), receipt).

    Each run is (start_node, end_node, entry_port_index, exit_port_index).  A
    hold shorter than minimum_frames is admissible only when it ends at the
    final node.  Raises ValueError for malformed input or callbacks, an
    exceeded workspace bound, or when no complete programme exists.
    '''
    if opportunity_episodes is not None or hold_credit is not None or claim_hold_cost is not None:
        return _opportunity_solve(nodes,entry_sources,exit_sources,start_cost=start_cost,
            hold_cost=hold_cost,cut_cost=cut_cost,start_allowed=start_allowed,final_cost=final_cost,
            final_allowed=final_allowed,minimum_frames=minimum_frames,check=check,
            workspace_limit_bytes=workspace_limit_bytes,episodes=opportunity_episodes,credit=hold_credit,
            claim_hold_cost=claim_hold_cost)
    nodes = np.asarray(nodes)
    if (nodes.ndim != 1 or len(nodes) < 2 or nodes.dtype.kind not in 'iu'
            or np.any(np.diff(nodes) <= 0)):
        raise ValueError('strictly increasing integer canonical nodes required')
    if (isinstance(minimum_frames, bool) or not isinstance(minimum_frames, (int, np.integer))
            or minimum_frames < 1):
        raise ValueError('positive integer minimum_frames required')
    N = len(nodes)
    entries = _ports(entry_sources, N, 'entry')
    exits = _ports(exit_sources, N, 'exit')
    start = _vector(start_cost, len(entries[0]), 'start_cost')
    begin = _mask(start_allowed, len(entries[0]), 'start_allowed')
    final = (np.zeros(len(exits[-1])) if final_cost is None
             else _vector(final_cost, len(exits[-1]), 'final_cost'))
    finish = _mask(final_allowed, len(exits[-1]), 'final_allowed')
    persistent = 24 * sum(map(len, entries)) + 40 * sum(map(len, exits))
    if persistent > workspace_limit_bytes:
        raise ValueError('boundary-port DP state exceeds workspace bound')
    in_p = [np.full(len(e), np.inf) for e in entries]
    in_c = [np.zeros(len(e), np.int64) for e in entries]
    in_back = [np.full(len(e), -1, np.int64) for e in entries]
    out_p = [np.full(len(x), np.inf) for x in exits]
    out_c = [np.zeros(len(x), np.int64) for x in exits]
    out_start = [np.full(len(x), -1, np.int64) for x in exits]
    out_entry = [np.full(len(x), -1, np.int64) for x in exits]
    out_code = [np.full(len(x), -1, np.int64) for x in exits]
    in_p[0] = np.where(begin, start, np.inf)
    receipt = dict(nodes=N, entry_ports=int(sum(map(len, entries))),
                   exit_ports=int(sum(map(len, exits))), hold_calls=0, cut_calls=0,
                   hold_arcs=0, cut_arcs=0, short_pairs_skipped=0,
                   unreachable_pairs_skipped=0, persistent_bytes=int(persistent),
                   peak_workspace_bytes=int(persistent),
                   workspace_limit_bytes=int(workspace_limit_bytes), arcs_stored=False,
                   objective='lexicographic(primary_seconds, cut_count)')

    def reserve(arcs):
        need = persistent + _ARC_BYTES * arcs
        if need > workspace_limit_bytes:
            raise ValueError('callback matrix exceeds workspace bound; no silent domain reduction')
        receipt['peak_workspace_bytes'] = max(receipt['peak_workspace_bytes'], int(need))

    for j in range(1, N):
        width = len(exits[j])
        for i in range(j):
            check()
            if nodes[j] - nodes[i] < minimum_frames and j != N - 1:
                receipt['short_pairs_skipped'] += 1
                continue
            if not width or not np.isfinite(in_p[i]).any():
                receipt['unreachable_pairs_skipped'] += 1
                continue
            shape = (len(entries[i]), width)
            reserve(shape[0] * width)
            value=hold_cost(i,j);codes=None
            if isinstance(value,(tuple,list)) and len(value)==3:
                value,codes=value[:2],np.asarray(value[2])
                if codes.shape!=shape or codes.dtype.kind not in 'iu':
                    raise ValueError('hold camera codes require a same-shaped integer matrix')
            cost, allowed = _matrix(value, shape, 'hold')
            if codes is not None and np.any(codes[allowed]<0):raise ValueError('admitted hold requires a frozen camera code')
            receipt['hold_calls'] += 1
            receipt['hold_arcs'] += int(cost.size)
            allowed = allowed & (entries[i][:, None] == exits[j][None, :])
            candidate = np.where(allowed, in_p[i][:, None] + cost, np.inf)
            row, low, fewest = _lexmin(candidate, np.broadcast_to(in_c[i][:, None], shape))
            better = np.isfinite(low) & ((low < out_p[j]) | ((low == out_p[j]) & (fewest < out_c[j])))
            out_p[j][better] = low[better]
            out_c[j][better] = fewest[better]
            out_start[j][better] = i
            out_entry[j][better] = row[better]
            out_code[j][better] = -1 if codes is None else codes[row[better],np.flatnonzero(better)]
        if j == N - 1 or not len(entries[j]) or not np.isfinite(out_p[j]).any():
            continue
        check()
        shape = (width, len(entries[j]))
        reserve(shape[0] * shape[1])
        cost, allowed = _matrix(cut_cost(j), shape, 'cut')
        receipt['cut_calls'] += 1
        receipt['cut_arcs'] += int(cost.size)
        allowed = allowed & (exits[j][:, None] != entries[j][None, :])
        candidate = np.where(allowed, out_p[j][:, None] + cost, np.inf)
        row, low, fewest = _lexmin(candidate, np.broadcast_to(out_c[j][:, None] + 1, shape))
        reached = np.isfinite(low)
        in_p[j] = np.where(reached, low, np.inf)
        in_c[j] = np.where(reached, fewest, 0)
        in_back[j] = np.where(reached, row, -1)
    total = np.where(finish, out_p[-1] + final, np.inf)
    if not np.isfinite(total).any():
        raise ValueError('no complete boundary-port programme')
    row, low, fewest = _lexmin(total[:, None], out_c[-1][:, None])
    exit_port, end, runs, camera_codes = int(row[0]), N - 1, [], []
    while True:
        first, entry_port = int(out_start[end][exit_port]), int(out_entry[end][exit_port])
        runs.append((first, end, entry_port, exit_port))
        camera_codes.append(int(out_code[end][exit_port]))
        if first == 0:
            break
        end, exit_port = first, int(in_back[first][entry_port])
    receipt['selected_arc_codes']=list(reversed(camera_codes))
    return tuple(reversed(runs)), (float(low[0]), int(fewest[0])), receipt


def _opportunity_solve(nodes,entries,exits,*,episodes,credit,claim_hold_cost=None,**kw):
    """One paid bit for the CURRENT nonoverlapping opportunity episode.

    Claims are optional, so a weak early view does not consume a better later
    one. A hold crossing an episode end forgets only that finished episode.
    Re-entry, A/B/A loops, and cloned source families cannot claim it twice.
    No all-history subset, cut quota, maximum hold, or source ban is added.
    """
    nodes=np.asarray(nodes);episodes=np.asarray(episodes)
    if (episodes.shape!=nodes.shape or episodes.dtype.kind not in 'iu'
            or np.any(np.diff(episodes)<0) or (not callable(credit) and not callable(claim_hold_cost))):
        raise ValueError('ordered fixed opportunity episodes and actual exposure credit required')
    if claim_hold_cost is not None and (not callable(claim_hold_cost) or credit is not None):
        raise ValueError('choose either legacy exposure credit or camera-dependent claimed holds')
    original_hold=kw.pop('hold_cost');original_cut=kw.pop('cut_cost')
    entries=_ports(entries,len(nodes),'entry');exits=_ports(exits,len(nodes),'exit')
    start=_vector(kw.pop('start_cost'),len(entries[0]),'start_cost')
    allowed=kw.pop('start_allowed');finish=kw.pop('final_allowed');final=kw.pop('final_cost')
    start_mask=_mask(allowed,len(entries[0]),'start_allowed')
    if final is not None:final=_vector(final,len(exits[-1]),'final_cost')
    if finish is not None:finish=_mask(finish,len(exits[-1]),'final_allowed')
    # At the beginning no event has been paid. The other bit is unreachable,
    # not a fake extra source/actor or a zero-cost quality measurement.
    expanded_start=np.repeat(start,2)
    expanded_allowed=np.column_stack((start_mask,np.zeros(len(start_mask),bool))).ravel()
    def alternatives(i,j):
        value=original_hold(i,j)
        base,valid=_matrix(value[:2],(len(entries[i]),len(exits[j])),'hold')
        codes=None if len(value)!=3 else np.asarray(value[2])
        if claim_hold_cost is None:
            benefit=np.asarray(credit(i,j),float)
            if benefit.shape==(len(entries[i]),):benefit=np.broadcast_to(benefit[:,None],base.shape)
            claimed=base-benefit;claim_valid=valid;claim_codes=codes
        else:
            claim=claim_hold_cost(i,j)
            if not isinstance(claim,(tuple,list)) or len(claim)!=4:
                raise ValueError('claimed hold requires cost, mask, frozen camera code and actual benefit')
            claimed,claim_valid=_matrix(claim[:2],base.shape,'claimed hold')
            claim_codes=np.asarray(claim[2]);benefit=np.asarray(claim[3],float)
            if (codes is None or claim_codes.shape!=base.shape or claim_codes.dtype.kind not in 'iu'
                    or np.any(claim_codes[claim_valid]<0) or np.any(claim_valid&~valid)):
                raise ValueError('claimed hold must preserve legal sources and actual frozen camera codes')
        if benefit.shape==(len(entries[i]),):benefit=np.broadcast_to(benefit[:,None],base.shape)
        if benefit.shape!=base.shape or not np.isfinite(benefit).all() or np.any(benefit<0):
            raise ValueError('finite nonnegative supported opportunity exposure required')
        return base,valid,codes,claimed,claim_valid,claim_codes,benefit
    def hold(i,j):
        base,valid,codes,claimed,claim_valid,claim_codes,benefit=alternatives(i,j)
        shape=(len(entries[i]),2,len(exits[j]),2)
        cost=np.zeros(shape);mask=np.zeros(shape,bool)
        frozen=None if codes is None else np.broadcast_to(codes[:,None,:,None],shape).copy()
        same=bool(episodes[i]==episodes[j])
        # Unpaid -> unpaid may defer the claim; unpaid -> paid may take it.
        cost[:,0,:,0]=base;mask[:,0,:,0]=valid
        if same:
            cost[:,0,:,1]=claimed;mask[:,0,:,1]=claim_valid&(benefit>0)
            if frozen is not None:frozen[:,0,:,1]=claim_codes
            cost[:,1,:,1]=base;mask[:,1,:,1]=valid
        else:
            # Crossing a fixed event resets its paid bit. The claimed
            # camera may cost more than the unclaimed camera, so claiming
            # must remain optional even when both map to the unpaid bit.
            take=claim_valid&(benefit>0)&((claimed<base)|~valid)
            cost[:,0,:,0]=np.where(take,claimed,base)
            mask[:,0,:,0]=valid|take
            if frozen is not None:frozen[:,0,:,0]=np.where(take,claim_codes,codes)
            cost[:,1,:,0]=base;mask[:,1,:,0]=valid
        result=(cost.reshape(shape[0]*2,shape[2]*2),mask.reshape(shape[0]*2,shape[2]*2))
        if frozen is not None:result=(*result,frozen.reshape(shape[0]*2,shape[2]*2))
        return result
    def cut(n):
        base,valid=_matrix(original_cut(n),(len(exits[n]),len(entries[n])),'cut')
        shape=(len(exits[n]),2,len(entries[n]),2)
        cost=np.broadcast_to(base[:,None,:,None],shape).reshape(shape[0]*2,shape[2]*2)
        mask=np.zeros(shape,bool);mask[:,0,:,0]=valid;mask[:,1,:,1]=valid
        return cost,mask.reshape(shape[0]*2,shape[2]*2)
    runs,objective,receipt=solve(nodes,[np.repeat(v,2) for v in entries],[np.repeat(v,2) for v in exits],
        start_cost=expanded_start,start_allowed=expanded_allowed,hold_cost=hold,cut_cost=cut,
        final_cost=None if final is None else np.repeat(final,2),
        final_allowed=None if finish is None else np.repeat(finish,2),**kw)
    paid=[]
    for k,(i,j,p,q) in enumerate(runs):
        base,valid,codes,claimed,claim_valid,claim_codes,benefit=alternatives(i,j)
        a,b=p//2,q//2;amount=float(benefit[a,b])
        take=bool(claim_valid[a,b] and amount>0 and (claimed[a,b]<base[a,b] or not valid[a,b]))
        claim=p%2==0 and (take if episodes[i]!=episodes[j] else q%2==1) and amount>0
        if claim and claim_codes is not None and receipt['selected_arc_codes'][k]!=int(claim_codes[a,b]):
            raise AssertionError('claimed camera differs from the camera frozen by DP')
        paid.append(dict(episode=int(episodes[i]),credit=amount if claim else 0.,claimed=bool(claim)))
    claimed=[p['episode'] for p in paid if p['claimed']]
    if len(set(claimed))!=len(claimed):raise AssertionError('one editorial opportunity paid twice')
    receipt.update(opportunity_claims=paid,current_event_paid_bit=True,all_history_subsets_stored=False,
        fixed_opportunity_episodes=len(np.unique(episodes)),opportunity_credit_once_only=True,
        claim_camera_chosen_before_argmin=claim_hold_cost is not None)
    return tuple((i,j,p//2,q//2) for i,j,p,q in runs),objective,receipt
