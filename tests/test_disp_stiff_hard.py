import numpy as np
import tensorflow as tf
import pytest
from pathlib import Path
from models import DISP_STIFF_UQPINN


@pytest.fixture()
def dummy_model():
    run_name = Path("test")
    model = DISP_STIFF_UQPINN(run_name, .9, 1.5, 1.)
    model.initialize_nns(1, 1, 1)
    return model


def test_left_boundary(mocker,dummy_model):
    locations = tf.cast(tf.zeros((10, 1)), dtype=tf.float32)
    Z = tf.cast(tf.ones((10, 1)), dtype=tf.float32)
    dummy_model.U_generator=mocker.Mock()
    dummy_model.U_generator.forward_pass.return_value=tf.ones_like(locations)

    assert dummy_model.get_left_boundary(locations, Z) == 1.0
