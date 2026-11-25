"""Run blocking tau2 environment steps outside the veRL worker event loop."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import cache, partial
import os


@cache
def _pool():
    workers = int(os.getenv("TAU2_ENV_THREADS", "0"))
    if workers == 0:
        return None
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tau2-env")


async def run_environment(operation, *args):
    executor = _pool()
    if executor is None:
        return operation(*args)

    call = partial(copy_context().run, operation, *args)
    pending = asyncio.get_running_loop().run_in_executor(executor, call)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(pending)
            break
        except asyncio.CancelledError:
            if pending.cancelled():
                raise
            cancelled = True
            # A running tool call can still mutate the session. Let it finish before cleanup.
    if cancelled:
        raise asyncio.CancelledError
    return result
