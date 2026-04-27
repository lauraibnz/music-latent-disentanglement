import pickle
import torch
import numpy as np


class AudioExample:
    """Container for audio data, arrays, and metadata - inspired by AFTER's design."""
    
    def __init__(self, output_type="torch"):
        self.output_type = output_type
        self.arrays = {}
        self.metadata = {}
    
    def put_array(self, key, array, dtype=None):
        """Add an array (audio, latents, etc.) to the example."""
        if torch.is_tensor(array):
            # Clone tensor inputs so sliced/views do not retain oversized shared storage.
            self.arrays[key] = array.detach().cpu().clone()
        else:
            array = np.asarray(array, dtype=dtype) if dtype else array
            self.arrays[key] = torch.from_numpy(array)
    
    def get(self, key, default=None, output_type=None):
        """Get an array with flexible output type (unified interface like AFTER)."""
        if key not in self.arrays:
            return default
        
        tensor = self.arrays[key]
        out_type = output_type or self.output_type
        
        if out_type == "torch":
            return tensor
        elif out_type == "numpy":
            return tensor.numpy()
        else:
            raise ValueError(f"output_type must be 'numpy' or 'torch', got {out_type}")
    
    def put_metadata(self, metadata):
        """Store metadata dictionary."""
        self.metadata = metadata.copy() if metadata else {}
    
    def get_metadata(self):
        """Get metadata dictionary."""
        return self.metadata
    
    def get_keys(self):
        """Get all available array keys."""
        return list(self.arrays.keys())
    
    def __repr__(self):
        array_info = {k: f"{v.dtype} {v.shape}" for k, v in self.arrays.items()}
        return f"AudioExample(arrays={array_info}, metadata_keys={list(self.metadata.keys())})"
    
    def to_bytes(self):
        """Serialize to bytes for LMDB storage."""
        return pickle.dumps({
            'arrays': self.arrays,
            'metadata': self.metadata,
            'output_type': self.output_type
        })
    
    @classmethod
    def from_bytes(cls, data):
        """Deserialize from bytes."""
        obj_dict = pickle.loads(data)
        example = cls(output_type=obj_dict.get('output_type', 'torch'))
        example.arrays = obj_dict.get('arrays', {})
        example.metadata = obj_dict.get('metadata', {})
        return example
