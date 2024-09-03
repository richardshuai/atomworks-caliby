import pytest
from torch.utils.data import ConcatDataset, Dataset, SequentialSampler

from datahub.samplers import DistributedMixedSampler, MixedSampler


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def dummy_datasets():
    data1 = list(range(100))
    data2 = list(range(1000, 1100))  # Different range for testing
    data3 = list(range(2000, 2100))  # Different range for testing
    dataset1 = DummyDataset(data1)
    dataset2 = DummyDataset(data2)
    dataset3 = DummyDataset(data3)
    return dataset1, dataset2, dataset3


def test_distributed_mixed_sampler(dummy_datasets):
    """
    Test that the DistributedMixedSampler correctly samples from the datasets with the specified probabilities,
    ensuring that each node gets a different slice of the dataset and that the distribution of samples reflects
    the specified probabilities.
    """
    dataset1, dataset2, dataset3 = dummy_datasets

    # Samplers
    sampler_1 = SequentialSampler(dataset1)
    sampler_2 = SequentialSampler(dataset2)

    datasets_info_1 = [
        {"sampler": sampler_1, "dataset": dataset1, "probability": 0.9},
        {"sampler": sampler_2, "dataset": dataset2, "probability": 0.1},
    ]
    # First mixed sampler
    datasets_1_2_concat = ConcatDataset([dataset1, dataset2])
    mixed_sampler = MixedSampler(datasets_info=datasets_info_1)

    # Second mixed sampler
    sampler_3 = SequentialSampler(dataset3)
    datasets_info_2 = [
        {"sampler": mixed_sampler, "dataset": datasets_1_2_concat, "probability": 0.5},
        {"sampler": sampler_3, "dataset": dataset3, "probability": 0.5},
    ]
    datasets_1_2_3_concat = ConcatDataset([datasets_1_2_concat, dataset3])
    dist_mixed_sampler_rank_0 = DistributedMixedSampler(
        datasets_info=datasets_info_2,
        n_examples_per_epoch=100,
        num_replicas=2,
        rank=0,
        shuffle=True,
    )

    dist_mixed_sampler_rank_1 = DistributedMixedSampler(
        datasets_info=datasets_info_2,
        n_examples_per_epoch=100,
        num_replicas=2,
        rank=1,
        shuffle=True,
    )

    indices_node_0 = list(dist_mixed_sampler_rank_0)
    indices_node_1 = list(dist_mixed_sampler_rank_1)

    assert len(indices_node_0) == 50
    assert len(indices_node_1) == 50

    # Ensure the slices are different
    assert set(indices_node_0).isdisjoint(set(indices_node_1))

    # Combine indices from both nodes to check distribution
    combined_indices = indices_node_0 + indices_node_1

    # Check if the distribution is close to the expected 90-10 ratio
    dataset1_count = sum(1 for idx in combined_indices if idx < 100)
    dataset2_count = sum(1 for idx in combined_indices if 100 <= idx < 200)
    dataset3_count = sum(1 for idx in combined_indices if idx >= 200)

    assert dataset1_count == 45  # 50% * 90% = 45
    assert dataset2_count == 5  # 50% * 10% = 5
    assert dataset3_count == 50  # 50% of the samples should be from dataset3

    # Load indices from the concat_dataset and ensure they are in the expected range
    for idx in indices_node_0[:10] + indices_node_1[:10]:  # Check a few indices from each node
        item = datasets_1_2_3_concat[idx]
        if idx < 100:
            assert 0 <= item < 100  # Should be in the range of dataset1
        elif idx < 200:
            assert 1000 <= item < 1100
        else:
            assert 2000 <= item < 2100


if __name__ == "__main__":
    pytest.main([__file__])
