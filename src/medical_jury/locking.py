from pathlib import Path


class RunnerLock:
    """Single writer/runner process; budget and report reads do not take this lock."""
    def __init__(self, root: Path):
        file = root / ".runtime/runner.lock"
        file.parent.mkdir(parents=True, exist_ok=True)
        self.stream = file.open("a+b")

    def __enter__(self):
        try:
            if __import__("os").name == "nt":
                import msvcrt
                self.stream.seek(0)
                if not self.stream.read(1):
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("Another Agent runner is active. Stop it before starting a second writer.") from None
        return self

    def __exit__(self, *args):
        self.stream.close()
