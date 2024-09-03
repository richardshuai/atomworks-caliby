import numpy as np

from datahub.utils.numpy import insert_data_by_id_, select_data_by_id


def test_select_data_by_id():
    to_ids = np.array([1, 5, 2, 20, 20, 2])

    from_array = np.arange(10).repeat(6).reshape(10, 6)
    from_ids = np.array([1, 2, 5, 6, 7, 21, 22, 23, 20, 25])

    solution = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    selected = select_data_by_id(to_ids, from_ids, from_array)

    assert np.all(selected == solution)


def test_insert_data_by_id():
    to_array = np.zeros((7, 6))
    to_ids = np.array([1, 5, 2, 17, 20, 20, 2])

    from_array = np.arange(10).repeat(6).reshape(10, 6)
    from_ids = np.array([1, 2, 5, 6, 7, 21, 22, 23, 20, 25])

    solution = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    insert_data_by_id_(to_array, to_ids, from_array, from_ids)

    assert np.all(to_array == solution)
