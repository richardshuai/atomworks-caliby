import pytest

from datahub.transforms.base import AddData, Compose, Identity, RandomRoute, RemoveKeys
from datahub.utils.rng import create_rng_state_from_seeds, rng_state

TEST_CASES = [
    {
        "seed": 43,
        "expected_transform_history": ["AddData", "RandomRoute", "AddData"],
        "expected_data": {"test": "value", "test3": "value3"},
    },
    {
        "seed": 1,
        "expected_transform_history": ["AddData", "RandomRoute", "RemoveKeys", "AddData"],
        "expected_data": {"test3": "value3"},
    },
    {
        "seed": 4,
        "expected_transform_history": ["AddData", "RandomRoute", "AddData", "AddData"],
        "expected_data": {"test": "value", "test2": "value2", "test3": "value3"},
    },
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_route_probabilistically(test_case):
    with rng_state(create_rng_state_from_seeds(np_seed=test_case["seed"])):
        pipe = Compose(
            [
                AddData(data={"test": "value"}),
                RandomRoute(
                    transforms=[Identity(), RemoveKeys(keys=["test"]), AddData({"test2": "value2"})],
                    probs=[0.3, 0.5, 0.2],
                ),
                AddData({"test3": "value3"}),
            ]
        )

        data = pipe({})

    history = [t["name"] for t in data.__transform_history__]
    assert history == test_case["expected_transform_history"]
    assert data == test_case["expected_data"]


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
