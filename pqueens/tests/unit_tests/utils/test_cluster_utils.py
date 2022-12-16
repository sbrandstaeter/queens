"""Unit tests of cluster utils."""
import pytest

from pqueens.utils.cluster_utils import distribute_procs_on_nodes_pbs

valid_testdata = [
    (1, 16, 1, 1),
    (3, 16, 1, 3),
    (32, 16, 2, 16),
    (18, 16, 2, 9),
    (1, 24, 1, 1),
    (3, 24, 1, 3),
    (26, 24, 2, 13),
]


@pytest.mark.parametrize(
    "num_procs, max_procs_per_node, num_nodes_expected, procs_per_node_expected", valid_testdata
)
def test_distribute_procs_on_nodes_pbs(
    num_procs, max_procs_per_node, num_nodes_expected, procs_per_node_expected
):
    """Test correct distribution of processors on the nodes of PBS culster."""
    num_nodes, procs_per_node = distribute_procs_on_nodes_pbs(
        num_procs=num_procs, max_procs_per_node=max_procs_per_node
    )

    assert num_nodes == num_nodes_expected
    assert procs_per_node == procs_per_node_expected


invalid_testdata = [
    (19, 16),
    (46, 16),
    (25, 24),
]


@pytest.mark.parametrize("num_procs, max_procs_per_node", invalid_testdata)
def test_fail_distribute_procs_on_nodes_pbs(num_procs, max_procs_per_node):
    """Test correct distribution of processors on the nodes of PBS culster."""
    with pytest.raises(ValueError):
        distribute_procs_on_nodes_pbs(num_procs=num_procs, max_procs_per_node=max_procs_per_node)
