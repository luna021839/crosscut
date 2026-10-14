from crosscut2.focal_output_correspondence import match,predictions_by_tick


def face(x=.5,y=.35,h=.2):return dict(center=[x,y],height=h,confidence=.99)


def test_missed_tilted_foreground_does_not_borrow_background_and_invent_size_jump():
    expected=face(h=.20833);background=face(.1894,.6769,.08241)
    selected,receipt=match([background],expected)
    assert selected is None and not receipt['matched']
    assert receipt['reason']=='detected_faces_do_not_match_planned_focal'
    assert receipt['candidates']==[background]


def test_geometric_focal_can_be_smaller_than_edge_face_without_being_identity_proof():
    a=face(.05,.4,.4);b=face(.5,.35,.2)
    selected,receipt=match([a,b],face())
    assert selected==1 and receipt['matched'] and not receipt['named_identity_certified']


def test_equal_plausible_output_detections_are_ambiguous_not_certified():
    selected,receipt=match([face(.49),face(.51)],face())
    assert selected is None and receipt['correspondence_ambiguous']


def test_overlapping_windows_prefer_exact_endpoint_or_report_conflicting_predictions():
    values=[face() for _ in range(4)];values[2]['face_native_sample_is_actual_endpoint']=True
    a=dict(output_frame=10,expected_output_focal=values)
    other=[face(.8) for _ in range(4)];other[0]['face_native_sample_is_actual_endpoint']=False
    b=dict(output_frame=17,expected_output_focal=other)
    predictions=predictions_by_tick([a,b],100)
    assert predictions[10]['height']==.2 and not predictions[10]['prediction_conflict']
