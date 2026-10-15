import copy
import numpy as np
import pytest
from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.focal_appearance import cosine
from crosscut2.played_partial_pose import features, matrix
from test_crosscut2_focal_activity_binding import two_faces


def catalogue():
    c=two_faces(first_moves=False)
    c.bank['vectors'][:]=0.;c.bank['vectors'][:,:,0,0]=1.;c.bank['vectors'][:,:,1,1]=1.
    return c


def sample(c, side, camera):
    return c.observation(1,side,np.tile(camera,(7,1)))


def test_crop_descriptor_belongs_to_its_selected_face_not_native_largest():
    c=catalogue()
    native=sample(c,0,[1.,.5,.5]);crop=sample(c,1,[1.35,.62,.5])
    assert native['head_index']==0 and crop['head_index']==1
    assert cosine(native,crop)==0.
    assert cosine(crop,sample(c,0,[1.35,.62,.5]))==1.
    assert not crop['appearance_vector'].flags.writeable


def test_fourth_and_missing_focal_do_not_borrow_descriptor_zero():
    c=catalogue();c.bank['heads'][:,:,3]=c.bank['heads'][:,:,1];c.bank['heads'][:,:,1]=0.
    known=sample(c,0,[1.,.5,.5]);fourth=sample(c,1,[1.35,.62,.5])
    assert fourth['head_index']==3 and np.isnan(cosine(known,fourth))
    c.bank['heads'][:]=0.;missing=sample(c,1,[1.,.5,.5])
    assert missing['head_index']==-1 and np.isnan(cosine(known,missing))


def test_descriptor_snapshot_does_not_follow_later_bank_mutation():
    c=catalogue();before=sample(c,0,[1.,.5,.5]);expected=before['appearance_vector'].copy()
    c.bank['vectors'][:]=0.
    assert np.array_equal(before['appearance_vector'],expected)


def test_reordered_head_and_descriptor_remain_the_same_physical_face():
    c=catalogue();old=sample(c,1,[1.35,.62,.5])
    bank=copy.deepcopy(c.bank);bank['heads'][:,:,0:2]=bank['heads'][:,:,0:2][:,:,::-1].copy()
    bank['vectors'][:,:,0:2]=bank['vectors'][:,:,0:2][:,:,::-1].copy()
    d=SourceCatalogue(c.nodes,bank,c.body,[],dict(presence=np.zeros((len(c.nodes),2,2))),[],fps=30.)
    new=sample(d,1,[1.35,.62,.5])
    assert old['head_index']==1 and new['head_index']==0 and cosine(old,new)==pytest.approx(1.)


def test_anonymous_appearance_now_changes_risk_but_not_geometry_admission():
    c=catalogue();a,b=sample(c,0,[1.,.5,.5]),sample(c,1,[1.,.5,.5])
    before=matrix(features([a],fps=30.),features([b],fps=30.))
    b['appearance_vector']=np.eye(128,dtype=np.float32)[1]
    after=matrix(features([a],fps=30.),features([b],fps=30.))
    assert cosine(a,b)==0.
    np.testing.assert_allclose(after[0]-before[0],.5)
    np.testing.assert_array_equal(before[1],after[1])
