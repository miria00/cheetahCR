"""Current-chip 13B retention and 70B GQA analytical controls; no optimizer.

Run with the existing Python 3.11 producer environment. Original source
obligations remain; this is a cost control, not a new physical admission.
"""
from pathlib import Path
from fractions import Fraction as F
from copy import deepcopy
import argparse
import hashlib
import json
import sys
import numpy as np

P = Path(__file__).resolve().parents[1]
S = P.parent/'systems/cheetah_joint'
sys.path.insert(0,str(S))
from wo22_dense_staging import dma, costs
from wo23_dense_banks_geometry_v2 import exact_minimum as exact25
from wo24_dense_banks import exact_minimum as exact31
from wo23_dense_mapping import alternative

ap=argparse.ArgumentParser()
ap.add_argument('--model',type=int,choices=[13,70],required=True)
a=ap.parse_args()
out=P/'update/paper_evidence_20261106/figure3_pies_v3'/f'controls_{a.model}'
out.mkdir(exist_ok=False)
bindings={}


def read(p,expected=None):
    p=Path(p);raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if expected:assert sha==expected,str(p)
    bindings[str(p)]=sha
    return json.loads(raw)


key='llama1_13b' if a.model==13 else 'llama3_70b'
r=read(P.parent/'Results/Blueprints/WO24/records'/(key+'.json'))
rp=Path(r['source_replay']['original_path'])
replay=read(rp,r['source_replay']['sha256'])
sp=Path(f'/tmp/cheetah_wo{24 if a.model==13 else 23}_20261106_v1/dense_lane_v1/{a.model}/mapping_source_v1/source.json')
s=read(sp,replay['bindings'][str(sp)])
h=r['hardware'];assert replay['passed'] and h==replay['hardware']
minimum=exact31 if a.model==13 else exact25
base=F(minimum(s,h)['decode_ns'])
assert base==F(r['latency_ns'])==F(replay['phase_ns']['decode'])
controls={}
delta=F();weight_count=0
for z in s['descriptors']:
    if not z['eligible_weight']:continue
    ext,stage=dma(z['address'],z['bytes'],h['host_interface_units'],h['staging_bytes'],h['staging_banks'])
    uses=int(z['decode_nonresident_count'])-int(z['decode_resident_count'])
    assert uses>=0
    delta+=uses*(F(ext+2*stage)+F(z['bytes'],h['bandwidth']))
    weight_count+=1
assert weight_count==replay['selected_weight_count']
controls['Weight residency']=dict(off_ns=str(base+delta),on_ns=str(base),saved_ns=str(delta),
    scope='Restore all original descriptor imports with setup, two staging passes and DRAM writes. Kernel weight reads and first imports stay paid; import constants are independent of mapping.')

if a.model==13:
    ep=Path('/tmp/cheetah_wo21_20261105_v1/dense/13/exchange_H15_v1/result.json')
    ex=read(ep);qp=Path(ex['source_proof_path']);q=read(qp)
    pp=Path(ex['prepared'])/'kv_gap_pages.npy';schp=qp.parent/'source_schedule.npz'
    for p,sha in [(pp,q['bindings'][str(pp)]),(schp,q['source_schedule_sha256'])]:
        digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==sha;bindings[str(p)]=digest
    pages=np.load(pp);sch=np.load(schp)
    mask=(~pages['terminal'])&(pages['read_phase']==1)
    addr=q['host_start']+sch['chain'][mask].astype(np.uint64)*q['slot_bytes']
    size=pages['bytes'][mask].astype(np.uint64);initial=sch['step'][mask]==0
    ext,stage=costs(addr,size,h['host_interface_units'],h['staging_bytes'],h['staging_banks'])
    rows=(size//(2*q['dim'])).astype(np.int64)
    terms=dict(host=F(sum(map(int,ext))),staging=2*F(sum(map(int,stage))),
        DRAM_write=F(sum(map(int,size)),h['bandwidth']),
        indexed_vector=F(sum(map(int,rows[~initial]))*(q['dim']+48),h['vector_lanes']),
        indexed_local=F(sum(map(int,rows[~initial]))*(12*q['dim']+96),h['local_GBps']))
    # Independent scalar chunk calendar checks at this chip's staging size.
    classes=np.unique(np.stack([addr%h['staging_bytes'],size],1),axis=0)
    for aa,nn in classes:
        aa,nn=int(aa),int(nn);first=min(nn,h['staging_bytes']-aa%h['staging_bytes']);left=nn-first
        parts=[(first,1),(h['staging_bytes'],left//h['staging_bytes']),
               (left%h['staging_bytes'],int(left%h['staging_bytes']!=0))]
        ee=ss=0
        for length,n in parts:
            if not n:continue
            words=(length+31)//32
            ee+=n*((words+h['host_interface_units']-1)//h['host_interface_units']+302)
            ss+=n*((words+h['staging_banks']-1)//h['staging_banks']+2)
        xx,yy=costs(np.array([aa],np.uint64),np.array([nn],np.uint64),h['host_interface_units'],h['staging_bytes'],h['staging_banks'])
        assert(ee,ss)==(xx[0],yy[0])
    delta=sum(terms.values(),F())
    controls['KV retention']=dict(off_ns=str(base+delta),on_ns=str(base),saved_ns=str(delta),
        source_reload_descriptors=int(mask.sum()),scalar_calendar_classes=len(classes),
        service_terms_ns={k:str(v)for k,v in terms.items()},
        scope='Original indexed KV reload contract on current WO24 hardware. All original cat work, initial writes and tails remain paid; same original head-major operands restored. Original lifetime proof inherited; extra imports and mapping costs rechecked.')
else:
    no_gqa=deepcopy(s);changed=0;invocations=0
    for old,z in zip(s['profiles'],no_gqa['profiles']):
        if z['original_mode']['implementation']!='private_GQA_head_rows':continue
        assert z['binding']['row']['matrix']['m']==1
        assert z['binding']['row']['matrix']['batches']==64
        # Group one is the original independent-head matrix response. It
        # retains the actual expanded operand owners and all original stores.
        z['original_mode']['gqa_group']=1
        z['options']=[alternative(z,m['matrix_tile'])for m in old['options']]
        assert z['binding']==old['binding'] and z['counts']==old['counts']
        for before,after in zip(old['options'],z['options']):
            assert before['matrix_tile']==after['matrix_tile']
            assert before['multiplicity']==after['multiplicity']==1
            assert before['reduction_cycles']==after['reduction_cycles']
        changed+=1;invocations+=z['counts'][1]
    assert changed==1022
    off=F(exact25(no_gqa,h)['decode_ns']);assert off>base
    controls['KV-head reuse (GQA)']=dict(off_ns=str(off),on_ns=str(base),saved_ns=str(off-base),
        original_shape_classes=changed,original_invocations=invocations,
        scope='Same-chip group-eight versus original independent-head group-one responses, reprice all 25 mappings and unchanged imports. Full original K order, owners, output coordinates and stores remain. This is KV-read reuse, not a KV-retention/offload control; semantic proof is inherited.')

for name in ['wo22_dense_staging.py','wo23_dense_mapping.py','wo24_dense_banks.py',
             'wo23_dense_banks_geometry_v2.py','wo14_gqa_candidate.py']:
    p=S/name;bindings[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(passed=True,workload=key,hardware=h,base_ns=str(base),controls=controls,
    eligible_weights=weight_count,bindings=bindings,optimizer_calls=0,
    source_admission='Inherited original source contracts; fresh full finite kernel and descriptor cost controls only. Not new full-model execution, quality or physical timing.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'model':a.model,'saved_seconds':{k:float(F(v['saved_ns'])/10**9)for k,v in controls.items()}}))
