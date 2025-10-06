from .audio_example import AudioExample

def save_to_lmdb(env, key, example):
    """Save AudioExample to LMDB."""
    with env.begin(write=True) as txn:
        txn.put(key.encode(), example.to_bytes())


def load_from_lmdb(env, key):
    """Load AudioExample from LMDB."""
    with env.begin() as txn:
        data = txn.get(key.encode())
        if data is None:
            return None
        return AudioExample.from_bytes(data)