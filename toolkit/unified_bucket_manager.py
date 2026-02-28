"""
Unified Bucket Manager for AI Toolkit

This module provides UnifiedBucketManager which combines buckets from multiple datasets
and creates unified batches with elements from different datasets mixed together.
"""

import random
import warnings
from typing import List, Dict, Tuple, TYPE_CHECKING
from toolkit.print import print_acc

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset


class UnifiedBucketManager:
    """
    Manages unified buckets across multiple datasets.
    
    Combines buckets with the same dimensions from different datasets and creates
    batches that mix elements from different datasets within the same bucket.
    
    Attributes:
        datasets: List of AiToolkitDataset instances
        batch_size: Number of items per batch
        unified_buckets: Dict mapping bucket_key to list of (dataset_idx, file_idx) tuples
        batch_indices: List of batches, each batch is a list of (dataset_idx, file_idx) tuples
    """
    
    def __init__(self, datasets: List['AiToolkitDataset'], batch_size: int):
        """
        Initialize UnifiedBucketManager.
        
        Args:
            datasets: List of AiToolkitDataset instances
            batch_size: Number of items per batch
        """
        self.datasets = datasets
        self.batch_size = batch_size
        # Maps bucket_key (e.g., "512x768") to list of (dataset_idx, local_file_idx) tuples
        self.unified_buckets: Dict[str, List[Tuple[int, int]]] = {}
        # Global batch indices for UnifiedBucketDataset
        # Each batch is a list of (dataset_idx, file_idx) tuples
        self.batch_indices: List[List[Tuple[int, int]]] = []
        
        # Validate dataset compatibility
        self.validate_dataset_compatibility()
    
    def validate_dataset_compatibility(self):
        """
        Validate that dataset configurations are compatible.
        
        Critical parameters (must match):
        - resolution
        - bucket_tolerance
        
        Warning parameters (should match, but not critical):
        - scale
        - random_scale
        - random_crop
        - augments
        - transforms
        """
        if len(self.datasets) <= 1:
            return
        
        # Get reference config from first dataset
        ref_config = self.datasets[0].dataset_config
        ref_resolution = ref_config.resolution
        ref_bucket_tolerance = ref_config.bucket_tolerance
        
        for idx, dataset in enumerate(self.datasets[1:], start=1):
            config = dataset.dataset_config
            
            # Critical checks - these must match
            if config.resolution != ref_resolution:
                raise ValueError(
                    f"Dataset {idx} has incompatible resolution: {config.resolution} "
                    f"(expected {ref_resolution}). All datasets must have the same resolution."
                )
            
            if config.bucket_tolerance != ref_bucket_tolerance:
                raise ValueError(
                    f"Dataset {idx} has incompatible bucket_tolerance: {config.bucket_tolerance} "
                    f"(expected {ref_bucket_tolerance}). All datasets must have the same bucket_tolerance."
                )
            
            # Warning checks - these should match but won't break functionality
            if config.scale != ref_config.scale:
                warnings.warn(
                    f"Dataset {idx} has different scale: {config.scale} "
                    f"(reference: {ref_config.scale}). Elements will use their own dataset's scale."
                )
            
            if config.random_scale != ref_config.random_scale:
                warnings.warn(
                    f"Dataset {idx} has different random_scale: {config.random_scale} "
                    f"(reference: {ref_config.random_scale}). Elements will use their own dataset's random_scale."
                )
            
            if config.random_crop != ref_config.random_crop:
                warnings.warn(
                    f"Dataset {idx} has different random_crop: {config.random_crop} "
                    f"(reference: {ref_config.random_crop}). Elements will use their own dataset's random_crop."
                )
    
    def build_unified_buckets(self, quiet=False):
        """
        Build unified buckets by combining buckets from all datasets.
        
        Iterates through all datasets and their buckets, combining buckets with
        the same bucket_key (dimensions). Stores (dataset_idx, local_file_idx) pairs
        for each element.
        
        Args:
            quiet: If True, suppress informational messages
        
        Example result:
            unified_buckets["512x768"] = [(0, 5), (0, 12), (1, 3), (1, 8), ...]
            where first number is dataset index, second is file index within that dataset
        """
        self.unified_buckets = {}
        
        for dataset_idx, dataset in enumerate(self.datasets):
            # Ensure dataset has buckets set up
            if not hasattr(dataset, 'buckets') or not dataset.buckets:
                raise ValueError(
                    f"Dataset {dataset_idx} ({dataset.dataset_path}) does not have buckets set up. "
                    f"Make sure buckets are enabled in dataset configuration."
                )
            
            # Iterate through all buckets in this dataset
            for bucket_key, bucket in dataset.buckets.items():
                # Create bucket if it doesn't exist in unified buckets
                if bucket_key not in self.unified_buckets:
                    self.unified_buckets[bucket_key] = []
                
                # Add all files from this bucket with (dataset_idx, file_idx) pairs
                for local_file_idx in bucket.file_list_idx:
                    self.unified_buckets[bucket_key].append((dataset_idx, local_file_idx))
        
        # Print unified bucket summary
        if not quiet:
            if len(self.datasets) > 1:
                print_acc(f'Unified bucket sizes ({len(self.datasets)} datasets combined):')
            else:
                print_acc(f'Unified bucket sizes for {self.datasets[0].dataset_path}:')
            for bucket_key, elements in self.unified_buckets.items():
                print_acc(f'  {bucket_key}: {len(elements)} files')
            print_acc(f'{len(self.unified_buckets)} buckets made')
    
    def shuffle_and_build_batches(self, quiet=False):
        """
        Shuffle elements within each bucket and build batches.
        
        Args:
            quiet: If True, suppress informational messages
        
        Process:
        1. For each bucket, shuffle all elements (mixing datasets)
        2. Create batches of size batch_size
        3. Shuffle the order of batches themselves
        4. Store in self.batch_indices
        """
        self.batch_indices = []
        
        # Process each bucket
        for bucket_key, elements in self.unified_buckets.items():
            # Shuffle elements within this bucket (mixes datasets)
            shuffled_elements = elements.copy()
            random.shuffle(shuffled_elements)
            
            # Create batches from shuffled elements
            for start_idx in range(0, len(shuffled_elements), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(shuffled_elements))
                batch = shuffled_elements[start_idx:end_idx]
                self.batch_indices.append(batch)
        
        # Shuffle the order of batches themselves
        random.shuffle(self.batch_indices)
        
        # Print batch creation summary
        if not quiet:
            print_acc(f'  - {len(self.batch_indices)} batches created (batch_size={self.batch_size})')
    
    def rebuild_for_epoch(self):
        """
        Rebuild batch_indices for a new epoch.
        
        Called by trigger_dataloader_setup_epoch() after datasets have shuffled
        their internal buckets. This rebuilds the unified buckets and creates
        new batch indices with the reshuffled data.
        
        Note: This is different from update_batch_size() - this method fully
        rebuilds buckets and batches for a new epoch, while update_batch_size()
        only recreates batches with a new size.
        """
        print_acc('Rebuilding unified buckets for new epoch')
        # Rebuild unified buckets (datasets have already shuffled their buckets)
        self.build_unified_buckets(quiet=True)
        # Create new batch indices with shuffled data
        self.shuffle_and_build_batches(quiet=True)
    
    def update_batch_size(self, new_batch_size: int):
        """
        Update batch_size and rebuild batch_indices without reshuffling buckets.
        
        This method is useful for runtime batch_size changes. It recreates batches
        with the new size but doesn't reshuffle the buckets themselves, preserving
        the current position in the epoch.
        
        Args:
            new_batch_size: New batch size to use
        """
        if self.batch_size == new_batch_size:
            return  # No change needed
        
        self.batch_size = new_batch_size
        # Recreate batches with new size (without reshuffling buckets)
        self.shuffle_and_build_batches()
