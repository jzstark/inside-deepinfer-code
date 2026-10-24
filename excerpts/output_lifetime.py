"""Run one inference step and return the generated outputs.

Returns:
    np.ndarray[np.uint64]: entrypoint -> a list of RequestOutput objects.

Note:
    The returned RequestOutput objects are managed by RequestOutputPool.
    RequestOutput will be automatically released in the next step() call.
    If you need to retain them beyond the current step, you must call pool.clone(list[RequestOutput]) to obtain copies,
    and release them via pool.release(list[RequestOutput]).
"""
