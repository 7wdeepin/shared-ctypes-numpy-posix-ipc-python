from ctypes import Structure, c_int32, c_int64


class MD(Structure):
    """Metadata for restoring frame
    """
    _fields_ = [
        ('height', c_int32),
        ('width', c_int32),
        ('channels', c_int32),
        ('size', c_int64)
    ]
