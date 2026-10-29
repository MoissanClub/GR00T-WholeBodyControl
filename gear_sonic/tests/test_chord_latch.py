from gear_sonic.utils.teleop.chord_latch import FullReleaseChordLatch


def test_chord_requires_complete_release_before_rearming():
    latch = FullReleaseChordLatch()

    assert latch.update((False, False, False, False)) is False
    assert latch.update((True, True, True, True)) is True
    assert latch.update((True, True, True, True)) is False

    # A transient dropped button and its return must not create another event.
    assert latch.update((True, True, False, True)) is False
    assert latch.update((True, True, True, True)) is False

    # Only a complete release rearms the chord.
    assert latch.update((False, False, False, False)) is False
    assert latch.update((True, True, True, True)) is True


def test_empty_chord_is_rejected():
    latch = FullReleaseChordLatch()

    try:
        latch.update(())
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("empty chord should be rejected")
