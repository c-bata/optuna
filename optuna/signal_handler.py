import signal
import sys
import types


def handler(signum: int, frame: types.FrameType) -> None:
    raise KeyboardInterrupt()


class handle_kill_signal:
    def __init__(self, study):
        self.study = study

    def __enter__(self):
        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGQUIT, _handle)
        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        return self

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGQUIT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        return False
