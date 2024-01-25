import logging
import os
import shutil as _shutil
import weakref as _weakref
from tempfile import mkdtemp


# class copied from tempfile and modified to
# not clean up when an exception is raised
# also can use KEEP_TEMP='yes' to not clean up at all
class TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    def __init__(self, suffix=None, prefix=None, dir=None):  # @ReservedAssignment
        self.name = mkdtemp(suffix, prefix, dir)
        self.autodelete = os.getenv("KEEP_TEMP", "no") == "no"
        if self.autodelete:
            self._finalizer = _weakref.finalize(
                self,
                self._cleanup,
                self.name,
                warn_message="Implicitly cleaning up {!r}".format(self),
            )

    @classmethod
    def _cleanup(cls, name, warn_message):
        _shutil.rmtree(name)
        logging.info(warn_message)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def __exit__(self, exc, value, tb):
        if exc is None:
            self.cleanup()
        logging.info(f"Leaving {self.name}")

    def cleanup(self):
        if self.autodelete and self._finalizer.detach():
            _shutil.rmtree(self.name)
