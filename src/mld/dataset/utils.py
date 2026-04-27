from .audio_example import AudioExample

def save_many_to_lmdb(env, entries):
    """Save a batch of ``(key, AudioExample)`` pairs to LMDB in one transaction."""
    if not entries:
        return
    with env.begin(write=True) as txn:
        for key, example in entries:
            txn.put(str(key).encode(), example.to_bytes())


def load_from_lmdb(env, key):
    """Load AudioExample from LMDB."""
    with env.begin() as txn:
        data = txn.get(key.encode())
        if data is None:
            return None
        return AudioExample.from_bytes(data)
