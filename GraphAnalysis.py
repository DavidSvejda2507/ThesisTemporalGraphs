import igraph as ig
import numpy as np
import warnings


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
            warnings.warn(
                UserWarning(
                    f"Partitions have different number of communities:\nPartition has {k_in} communities\nReference has {k_out} communities"
                )
            )
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
        if (key not in inputs) or (value not in outputs):
            warnings.warn(UserWarning("Accuracy mapping is nontrivial"))
        else:
            mapping[key] = value
            inputs.remove(key)
            outputs.remove(value)
            i -= 1
        contingencyTable[key, value] = 0

    correct = 0
    for input, output in zip(partition, reference):
        if mapping.get(input, None) == output:
            correct += 1

    return correct / n, mapping
