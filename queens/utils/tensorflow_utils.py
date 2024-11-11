"""Utils related to tensorflow and friends."""

import logging
import os

_logger = logging.getLogger(__name__)


def configure_tensorflow(tensorflow):
    """Configure tensorflow.

    Args:
        tensorflow (module): The module to configure
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tensorflow.get_logger().setLevel(logging.ERROR)

    # Use GPU acceleration if possible
    if tensorflow.test.gpu_device_name() != "/device:GPU:0":
        _logger.info("WARNING: GPU device not found.")
    else:
        _logger.info("SUCCESS: Found GPU: %s", tensorflow.test.gpu_device_name())


def configure_keras(tf_keras):
    """Configure tf keras.

    Args:
        tf_keras (Model): The module configuration
    """
    tf_keras.backend.set_floatx("float64")
