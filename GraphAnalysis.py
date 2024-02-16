import igraph as ig
import numpy as np
import warnings
import numba


def calculateAccuracy(partition, reference, k=None):
    """Try to match a partition to a refference partition such
    that the number of differing vertices is minimised

    Args:
        partition (list[int]): partition to be matched
        reference (list[int]): partition to be matched against
        k (int): number of communities to match (default to as many as possible)

    Returns:
        accuracy, map: accuracy between 0 and 1, mapping from partition to reference
    """
    n = len(reference)
    k_in = max(partition) + 1
    k_out = max(reference) + 1
    if k is None:
        if k_in != k_out:
            warnings.warn(f"Partitions have different number of communities:\nPartition has {k_in} communities\nReference has {k_out} communities")
        k = min(k_in, k_out)

    contingencyTable = np.zeros((k_in, k_out))
    for comm, comm_ in zip(partition, reference):
        contingencyTable[comm, comm_] += 1

    mapping = {}
    inputs = set(range(k_in))
    outputs = set(range(k_out))
    i = k
    while i > 0:
        key, value = np.unravel_index(contingencyTable.argmax(), (k_in, k_out))
        if (key in inputs) and (value in outputs):
            mapping[key] = value
            inputs.remove(key)
            outputs.remove(value)
            i -= 1
        contingencyTable[key, value] = -1

    correct = 0
    for input, output in zip(partition, reference):
        if mapping.get(input, None) == output:
            correct += 1

    return correct / n, mapping


from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


@numba.jit(nopython=True)
def Consistency(partition1, partition2):
    """Calculates the consistency of the partitions, assumes that the communities are numbred 0 through k

    Args:
        partition1 (iterable): list of integers specifying community membership
        partition2 (iterable): list of integers specifying community membership

    Returns:
        double: Consistency value
    """
    assert len(partition1) == len(partition2)
    vs = len(partition1)
    k1 = max(partition1) + 1
    k2 = max(partition2) + 1
    contingencyTable = np.zeros((k1, k2))
    for comm, comm_ in zip(partition1, partition2):
        contingencyTable[comm, comm_] += 1

    sum = 0
    for row in contingencyTable:
        rowsum = 0
        for n in row:
            sum += rowsum * n
            rowsum += n
    for col in contingencyTable.transpose():
        colsum = 0
        for n in col:
            sum += colsum * n
            colsum += n
    return 1 - 2 * sum / (vs * (vs - 1))


def ConsistencyCheck(partition1, partition2):
    assert len(partition1) == len(partition2)
    vs = len(partition1)
    sum = 0
    for i in range(vs):
        for j in range(i + 1, vs):
            if partition1[i] == partition1[j] and partition2[i] != partition2[j]:
                sum += 1
            if partition1[i] != partition1[j] and partition2[i] == partition2[j]:
                sum += 1
    return 1 - 2 * sum / (vs * (vs - 1))

def evaluatePartitions(graphs, partitions):
    mod_sum = 0
    cons_sum = 0
    for i in len(graphs):
        modularity = graphs[i].modularity(partitions[i])
        # print(f"Modularity on G{i}: {modularity}")
        mod_sum += modularity
        if i > 0:
            cons_sum += Consistency(partitions[i], partitions[i - 1])
    return mod_sum, cons_sum