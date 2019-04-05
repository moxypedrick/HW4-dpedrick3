import time
import mmap
from struct import unpack


def pagerank(index_file, edge_file, max_node_id, edge_count, damping_factor=0.85, iterations=10):
    index_map = mmap.mmap(
        index_file.fileno(),
        length=(max_node_id+1)*8,
        access=mmap.ACCESS_READ)

    edge_map = mmap.mmap(
        edge_file.fileno(),
        length=edge_count*8,
        access=mmap.ACCESS_READ)

    scores = [1.0 / (max_node_id + 1)] * (max_node_id + 1)

    start_time = time.time()

    for it in range(iterations):
        new_scores = [0.0] * (max_node_id + 1)
        for i in range(edge_count):
            source, target = unpack(
                '>i i',
                edge_map[i * 8: i * 8 + 8])
            source_degree = unpack(
                '<i i',
                index_map[source * 8: source * 8 + 8])[1]
            new_scores[target] += damping_factor * scores[source] / source_degree

        min_pr = (1 - damping_factor) / (max_node_id + 1)
        new_scores = [min_pr + item for item in new_scores]
        scores = new_scores

        print ("Completed {0}/{1} iterations. {2} seconds elapsed." \
            .format(it + 1, iterations, time.time() - start_time))

    print ()

    return scores