# Also, there's a timeout error which is managed by subprocess module.
class ModeledError(Exception):
    """
    This is a base class for all modeled errors.
    """
    pass


class CrashError(ModeledError):
    pass


class IncorrectResult(ModeledError):
    pass


class NaNError(ModeledError):
    pass


class PerfDegradation(ModeledError):
    pass


class RuntimeFailure(ModeledError):
    pass

# Timeout...


class MaybeDeadLoop(ModeledError):
    pass


class NNSmithInternalError(Exception):
    pass
