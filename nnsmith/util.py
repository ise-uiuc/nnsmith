import multiprocessing
import os
import sys
from contextlib import contextmanager
from multiprocessing import Pool, Queue, Process
import multiprocessing.pool


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    # credit: https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    if stdout is None:
        stdout = sys.stdout
    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def merged_stderr_stdout():  # $ exec 2>&1
    return stdout_redirected(to=sys.stdout, stdout=sys.stderr)


def forkpool_execution(func):
    '''func must be a reliable function that won't hang'''
    def server(q, rq):
        while True:
            args, kwargs = q.get()
            try:
                res = func(*args, **kwargs)
                rq.put(('res', res))
            except Exception as e:
                rq.put(('error', str(e)))
    pool = None
    q = None
    rq = None
    manager = multiprocessing.Manager()

    def wrapper(*args, **kwargs):
        nonlocal pool, q, rq
        if pool is None:
            q, rq = manager.Queue(), manager.Queue()
            pool = Process(target=server, args=(q, rq))
            pool.start()
        q.put((args, kwargs))
        ret = rq.get()
        if ret[0] == 'error':
            raise RuntimeError(ret[1])
        return ret[1]

    def terminate():
        nonlocal pool
        if pool is not None:
            pool.terminate()
            pool.join()

    wrapper.terminate = terminate
    return wrapper
