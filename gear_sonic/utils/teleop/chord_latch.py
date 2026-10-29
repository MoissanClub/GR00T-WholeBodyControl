"""Button-chord edge helpers for teleoperation controls."""

from collections.abc import Sequence


class FullReleaseChordLatch:
    """Emit one event per full chord press and rearm after complete release."""

    def __init__(self) -> None:
        self._latched = False

    def update(self, buttons_pressed: Sequence[bool]) -> bool:
        if not buttons_pressed:
            raise ValueError("buttons_pressed must not be empty")

        pressed = tuple(bool(value) for value in buttons_pressed)
        if not any(pressed):
            self._latched = False

        if all(pressed) and not self._latched:
            self._latched = True
            return True

        return False
