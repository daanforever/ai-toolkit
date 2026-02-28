"""
Unified Bucket Dataset for AI Toolkit

This module provides UnifiedBucketDataset which wraps multiple datasets and
uses UnifiedBucketManager to provide batches that mix elements from different datasets.
"""

import random
from typing import List, TYPE_CHECKING
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.unified_bucket_manager import UnifiedBucketManager


class UnifiedBucketDataset(Dataset):
    """
    Dataset wrapper for unified bucket management.
    
    This dataset uses UnifiedBucketManager to provide batches that contain
    elements from multiple datasets mixed together within the same bucket
    (same dimensions).
    
    Attributes:
        datasets: List of AiToolkitDataset instances
        bucket_manager: UnifiedBucketManager instance that manages unified buckets
    """
    
    def __init__(self, datasets: List['AiToolkitDataset'], bucket_manager: 'UnifiedBucketManager'):
        """
        Initialize UnifiedBucketDataset.
        
        Args:
            datasets: List of AiToolkitDataset instances
            bucket_manager: UnifiedBucketManager instance with built buckets and batches
        """
        self.datasets = datasets
        self.bucket_manager = bucket_manager
        self.len = None  # Cache for length, reset when buckets are rebuilt
    
    def __len__(self):
        """
        Return the number of batches.
        
        Returns:
            Number of batches in bucket_manager.batch_indices
        """
        if self.len is None:
            self.len = len(self.bucket_manager.batch_indices)
        return self.len
    
    def __getitem__(self, item):
        """
        Get a batch of items.
        
        For each (dataset_idx, file_idx) pair in the batch, retrieves the item
        from the corresponding dataset using _get_single_item().
        
        Args:
            item: Batch index
            
        Returns:
            List of FileItemDTO objects for the batch
        """
        # Handle case where index is out of range (can happen during epoch transitions)
        if len(self.bucket_manager.batch_indices) - 1 < item:
            # Pick another random index as fallback
            item = random.randint(0, len(self.bucket_manager.batch_indices) - 1)
        
        # Get the batch (list of (dataset_idx, file_idx) tuples)
        batch_indices = self.bucket_manager.batch_indices[item]
        
        # Retrieve items from their respective datasets
        batch_items = []
        for dataset_idx, file_idx in batch_indices:
            dataset = self.datasets[dataset_idx]
            # Use the dataset's _get_single_item method to get the processed item
            item_dto = dataset._get_single_item(file_idx)
            batch_items.append(item_dto)
        
        return batch_items
