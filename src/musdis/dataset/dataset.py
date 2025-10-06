import lmdb
from torch.utils.data import Dataset
from .utils import load_from_lmdb

class BaseDataset(Dataset):
    """Dataset for loading preprocessed audio chunks from LMDB (lazy env init per worker)."""
    
    def __init__(self, lmdb_path, transform=None, return_latents=True):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.return_latents = return_latents
        
        # Don't open LMDB here! Each worker will open its own environment
        self.env = None

        # Get length (open temporarily in main process)
        with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                self.length = txn.stat()['entries']

    def _init_env(self):
        """Lazily open LMDB environment inside each worker process."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,  # safer for multiple workers
                meminit=False
            )

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Ensure environment is initialized (works for multiprocessing)
        self._init_env()

        # Load AudioExample from LMDB
        key = f"{idx:08d}"
        example = load_from_lmdb(self.env, key)
        
        if example is None:
            raise IndexError(f"No data found for index {idx}")
        
        result = {'metadata': example.get_metadata()}
        
        # Get audio if available
        audio = example.get('audio', None)
        if audio is not None:
            if self.transform:
                audio = self.transform(audio)
            result['audio'] = audio
        
        # Add latents if available and requested
        if self.return_latents:
            latent = example.get('z', None)
            if latent is not None:
                result['latent'] = latent
        
        return result

    def __del__(self):
        """Ensure LMDB environment is closed."""
        if self.env is not None:
            self.env.close()
            self.env = None
