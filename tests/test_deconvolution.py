import pytest

from lionelmssq.deconvolution import create_averagine

AVERAGINE = {
    "none": {"C": 9.5, "H": 12.75, "N": 3.75, "O": 5.0, "P": 0.0},
    "phosphate": {"C": 9.5, "H": 12.75, "N": 3.75, "O": 7.0, "P": 1.0},
    "phosphate+": {"C": 9.5, "H": 12.75, "N": 3.75, "O": 7.0, "P": 1.0, "S": 0.0},
    "thiophosphate": {"C": 9.5, "H": 12.75, "N": 3.75, "O": 6.0, "P": 1.0, "S": 1.0},
}


@pytest.mark.parametrize("phosphate", [True, False])
@pytest.mark.parametrize("thiophosphate", [True, False])
def test_create_averagine(phosphate, thiophosphate):
    averagine = create_averagine(
        with_backbone=phosphate, with_thiophosphate_backbone=thiophosphate
    )

    testcase = "none"
    if thiophosphate:
        testcase = "thiophosphate"
    if phosphate:
        testcase = "phosphate+" if thiophosphate else "phosphate"
    print(averagine)
    print(AVERAGINE[testcase])
    assert averagine == AVERAGINE[testcase]
