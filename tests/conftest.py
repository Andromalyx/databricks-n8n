import pytest

from wealth_prediction.config import Config


@pytest.fixture
def game_config():
    return Config().game
