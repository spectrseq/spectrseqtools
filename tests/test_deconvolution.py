import pytest

from lionelmssq.deconvolution import create_averagine

AVERAGINE = {
    "no_backbone": {"C": 9.5, "H": 12.75, "N": 3.75, "O": 5.0, "P": 0.0, "S": 0.0},
    "phosphate": {"C": 9.5, "H": 12.75, "N": 3.75, "O": 7.0, "P": 1.0, "S": 0.0},
    "thiophosphate": {"C": 9.5, "H": 12.75, "N": 3.75, "O": 6.0, "P": 1.0, "S": 1.0},
}


@pytest.mark.parametrize("backbone", AVERAGINE.keys())
def test_create_averagine(backbone):
    averagine = create_averagine(backbone=backbone)
    print(averagine)
    print(AVERAGINE[backbone])
    assert averagine == AVERAGINE[backbone]
