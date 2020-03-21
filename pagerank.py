import argparse
import pathlib
import pickle
import time
from os import linesep

import numpy as np

import util
from graph import Links


class PageRank(object):
    def __init__(self, links):
        self._links = links
        self._stable_dist = None

    def compute(self, method, markov):
        # Transform IDs to integers
        nodes_list = list(self._links.ins.keys())
        n = len(nodes_list)
        id_by_node = {node: _id for _id, node in enumerate(nodes_list)}
        in_links = {id_by_node[node]: set(id_by_node[from_node] for from_node in from_nodes)
                    for node, from_nodes in self._links.ins.items()}
        # Note: only count the outgoing links which are parts of the graph.
        # e.x. In the case of a web crawl, out going links which are not crawled are ignored.
        out_links_count = {
            id_by_node[node]: len(list(filter(lambda x: x in self._links.ins, self._links.outs[node])))
            for node in self._links.outs.keys()
        }
        raw_stable_dist = method(in_links, out_links_count, n, markov)
        self._stable_dist = list(zip(nodes_list, raw_stable_dist))

    def get_score(self, node):
        return

    def get_total_score(self):
        return sum(score for _, score in self._stable_dist)

    def top(self, size=500):
        sorted_nodes = sorted(self._stable_dist, key=lambda t: t[1], reverse=True)
        return sorted_nodes[:size]


class PageRankMethod(object):
    """
    Suitable for a small and dense graph.
    """

    @classmethod
    def iterative_update(cls, init_dist, tolerance, update_dist_hook, max_iter):
        def check_convergence(old, new):
            diff = old - new
            boolean_filters = [value <= tolerance for value in diff]
            print(f"[INFO] Need {boolean_filters.count(False)} more nodes to converge")
            return boolean_filters.count(False) == 0

        cur_iter = 0
        interim_dist = init_dist
        while True:
            cur_iter += 1
            print(f"[INFO] Running at iteration {cur_iter}")
            if cur_iter > max_iter:
                print(f"[INFO] Reached the maximum iteration without seeing convergence "
                      f"(max_iter={max_iter}, tolerance={tolerance})")
                break

            old_dist = interim_dist
            interim_dist = update_dist_hook(interim_dist)

            if check_convergence(old_dist, interim_dist):
                print(f"[INFO] See convergence")
                break
        return interim_dist

    @classmethod
    def transition_matrix(cls, in_links, out_links_count, n, d, max_iter=100):
        L = np.zeros((n, n), dtype=np.bool_)
        for node, from_nodes in in_links.items():
            for from_node in from_nodes:
                L[node, from_node] = True

        M_inv_raw = [0] * n
        for node, num_of_out_links in out_links_count.items():
            M_inv_raw[node] = 1 / num_of_out_links if num_of_out_links != 0 else 0
        M_inv = np.array(M_inv_raw)

        transition = d * L * M_inv + ((1 - d) / n)

        tolerance = (1 / (n * 100))
        page_rank = np.array([1 / n] * n)

        def update_dist(dist):
            return transition.dot(dist)

        page_rank = cls.iterative_update(page_rank, tolerance, update_dist, max_iter)

        if page_rank is None:
            print("[ERROR] Couldn't find an eigenvalue of 1 for the transition matrix.")
            return []
        return page_rank

    @classmethod
    def algebraic(cls, in_links, out_links_count, n, d, max_iter=100):
        sink_nodes = {node for node, out_nodes_count in out_links_count.items()
                      if out_nodes_count == 0}

        tolerance = (1 / (n * 100))
        page_rank = np.array([1 / n] * n)

        def update_dist(dist):
            sink_page_rank = sum(dist[node] for node in sink_nodes)
            pre_sum = sum(dist)
            new_dist = []
            for _id, value in enumerate(dist):
                new_value = (1 - d) / n
                new_value += d * sink_page_rank / n
                for parent_id in in_links[_id]:
                    new_value += d * dist[parent_id] / out_links_count[parent_id]
                new_dist.append(new_value)
            cur_sum = sum(new_dist)
            return np.array(new_dist)

        return cls.iterative_update(page_rank, tolerance, update_dist, max_iter)


class Loader(object):
    @staticmethod
    def load_from_file(args) -> Links:
        link_graph_path = pathlib.Path(args.link_graph)
        if link_graph_path.is_file():
            with link_graph_path.open("rb") as fp:
                return pickle.load(fp)
        else:
            raise ValueError(f"{link_graph_path} is not a file.")

    @staticmethod
    def load_from_elastic_search(args) -> Links:
        return


def page_rank_file_writer(scored_pages, in_links, out_links, path):
    with path.open("w") as fp:
        for _id, page_rank in scored_pages:
            fp.write("\t".join([
                _id,
                str(page_rank),
                str(len(out_links[_id])),
                str(len(in_links[_id])),
            ]))
            fp.write(linesep)


def create_args_parser():
    parser = argparse.ArgumentParser(description='Link graph generation')
    parser.add_argument("link_graph", metavar="link-graph", action="store",
                        help="path of the link-graph pickle file")
    return parser


def main():
    parser = create_args_parser()
    args = parser.parse_args()

    print(f"[INFO] Start loading links...")
    load_file_start_t = time.time()
    links = Loader.load_from_file(args)
    print(f"[INFO] Links loaded, time_elapsed={util.time_used(load_file_start_t)}")

    print("[INFO] Initialize a PageRank object with loaded links.")
    init_pr_obj_start_t = time.time()
    pr = PageRank(links)
    print(f"[INFO] PageRank object is initialized, time_elapsed={util.time_used(init_pr_obj_start_t)}")

    print("[INFO] Start to compute PageRank for each node...")
    compute_pr_start_t = time.time()
    pr.compute(method=PageRankMethod.algebraic, markov=0.85)
    print(f"[INFO] PageRank computed, time_elapsed={util.time_used(compute_pr_start_t)}")
    print(f"[INFO] Total score is {pr.get_total_score()}")

    link_graph_filepath = pathlib.Path(args.link_graph)
    result_file_path = (link_graph_filepath.parent
                        .joinpath("results")
                        .joinpath(f"{link_graph_filepath.stem}_page_rank.txt"))
    page_rank_file_writer(pr.top(), links.ins, links.outs, result_file_path)
    print(f"[INFO] Result file is written at {result_file_path}")


if __name__ == '__main__':
    main()
