import pytest

from datahub.transforms.base import Compose, Transform, TransformPipelineError


class Transform1(Transform):
    incompatible_previous_transforms = ["Transform2"]

    def check_input(self, data):
        pass

    def forward(self, data):
        return data


class Transform2(Transform):
    def check_input(self, data):
        pass

    def forward(self, data):
        return data


class Transform3(Transform):
    def check_input(self, data):
        pass

    def forward(self, data):
        return data


class Transform4(Transform):
    requires_previous_transforms = ["Transform1", "Transform2", "Transform3"]
    previous_transforms_order_matters = True

    def check_input(self, data):
        pass

    def forward(self, data):
        return data


# Test all 4 cases
data = {"data": "data"}


def test_incompatible_previous_transforms():
    with pytest.raises(TransformPipelineError):
        transform = Compose([Transform2(), Transform1()], track_rng_state=False)
        transform(data)


def test_missing_previous_transform():
    with pytest.raises(TransformPipelineError):
        transform = Compose([Transform1(), Transform4()], track_rng_state=False)
        transform(data)


def test_wrong_order_previous_transforms():
    with pytest.raises(TransformPipelineError):
        transform = Compose([Transform3(), Transform1(), Transform2(), Transform4()], track_rng_state=False)
        transform(data)


def test_success():
    transform = Compose([Transform1(), Transform2(), Transform3(), Transform4()], track_rng_state=False)
    transform(data)


if __name__ == "__main__":
    # For interactive debugging
    pytest.main(["-v", "-x", __file__])
