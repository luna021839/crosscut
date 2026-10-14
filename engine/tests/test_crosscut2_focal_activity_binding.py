import copy
import numpy as np
from crosscut2.display_source_catalogue import SourceCatalogue
from test_crosscut2_event_holds import inputs


def two_faces(*,first_moves):
    nodes,banks,poses,styles=inputs(count=1)
    bank=copy.deepcopy(banks[0]);pose=poses[0].copy()
    bank['heads'][:,:,0]=[.30,.3,.25,.95]
    bank['heads'][:,:,1]=[.75,.3,.25,.95]
    stationary=pose[1,1,0].copy()
    pose[:,:,0]=stationary;pose[:,:,1]=stationary
    pose[:,:,1,:,0]+=.25
    if first_moves:pose[:,3,0,2:6,1]+=.08
    return SourceCatalogue(nodes,bank,pose,[],styles[0],[],fps=30.)


def test_offscreen_person_activity_cannot_reward_a_different_displayed_face():
    still=two_faces(first_moves=False);moving=two_faces(first_moves=True)
    camera=np.tile([1.35,.62,.5],(2,1));frames=np.array([18,19])
    assert moving.observation(1,1,np.tile(camera[0],(7,1)))['head_index']==1
    # Source face0 moves. The crop displays stationary face1 instead.
    np.testing.assert_allclose(moving.rates(frames,camera),still.rates(frames,camera),rtol=0,atol=1e-12)


def test_the_same_displayed_moving_person_still_receives_own_activity_utility():
    still=two_faces(first_moves=False);moving=two_faces(first_moves=True)
    camera=np.tile([1.,.5,.5],(2,1));frames=np.array([18,19])
    assert moving.observation(1,1,np.tile(camera[0],(7,1)))['head_index']==0
    assert (moving.rates(frames,camera)<still.rates(frames,camera)).all()


def test_batched_camera_choices_use_their_own_selected_face_activity():
    still=two_faces(first_moves=False);moving=two_faces(first_moves=True)
    frames=np.array([18,19]);camera=np.array([[[1.,.5,.5]]*2,[[1.35,.62,.5]]*2])
    a,b=moving.rates(frames,camera),still.rates(frames,camera)
    assert (a[0]<b[0]).all()
    np.testing.assert_allclose(a[1],b[1],rtol=0,atol=1e-12)
