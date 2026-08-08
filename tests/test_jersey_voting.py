from scout.perception.jersey import Read, vote_jerseys


def test_majority_wins_over_noise():
    reads = [Read(1, 7, 0.9, 5000)] * 8 + [Read(1, 1, 0.9, 5000)] * 2  # OCR reads "7" as "1" sometimes
    assert vote_jerseys(reads)[1] == 7


def test_small_crops_carry_less_weight():
    # 3 confident big-crop reads of 9 beat 6 tiny-crop reads of 8
    reads = [Read(1, 9, 0.9, 10000)] * 3 + [Read(1, 8, 0.9, 100)] * 6
    assert vote_jerseys(reads)[1] == 9


def test_ambiguous_track_gets_no_number():
    reads = [Read(1, 7, 0.9, 5000)] * 5 + [Read(1, 4, 0.9, 5000)] * 5
    assert 1 not in vote_jerseys(reads)  # 50/50 split < min_margin


def test_invalid_numbers_ignored():
    reads = [Read(1, 0, 0.9, 5000)] * 5 + [Read(1, 777, 0.9, 5000)] * 5
    assert vote_jerseys(reads) == {}


def test_tracks_independent():
    reads = [Read(1, 7, 0.9, 5000)] * 4 + [Read(2, 10, 0.9, 5000)] * 4
    v = vote_jerseys(reads)
    assert v == {1: 7, 2: 10}
