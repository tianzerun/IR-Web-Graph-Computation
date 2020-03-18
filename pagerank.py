import argparse
import pathlib

import numpy as np

from graph import Links, load_link_graph


class PageRank(object):
    def __init__(self, links: Links):
        self._links = links
        self._nodes = list(self._links.ins.keys())
        self._n = len(self._nodes)
        self._E = np.ones((self._n, self._n))
        self._L = np.array([[1 if from_node in self._links.ins[to_node] else 0 for from_node in self._nodes]
                            for to_node in self._nodes])
        self._M = np.diag([len(self._links.outs[node]) for node in self._nodes])
        self._stable_dist = None

    def _create_transition_matrix(self, d):
        return d * self._L.dot(np.linalg.inv(self._M)) + ((1 - d) / self._n) * self._E

    def compute(self, markov=0.85):
        transition = self._create_transition_matrix(d=markov)
        eigenvalues, eigenvectors = np.linalg.eig(transition)

        for i, value in enumerate(eigenvalues):
            if round(value, 4) == 1:
                stable = eigenvectors[:, i]
                normalization = sum(stable.real)
                self._stable_dist = zip(self._nodes, [round(value / normalization, 4) for value in stable.real])

        if self._stable_dist is None:
            print("[ERROR] Couldn't find an eigenvalue of 1 for the transition matrix.")
            return False
        return True

    def get_score(self, link):
        return

    def top(self, size=500):
        sorted_nodes = sorted(self._stable_dist, key=lambda t: t[1], reverse=True)
        return sorted_nodes[:size]


def create_args_parser():
    parser = argparse.ArgumentParser(description='Link graph generation')
    parser.add_argument("link_graph", metavar="link-graph", action="store",
                        help="path of the link-graph pickle file")
    return parser


class Loader(object):
    @staticmethod
    def load_from_file(args) -> Links:
        link_graph_path = pathlib.Path(args.link_graph)
        if link_graph_path.is_file():
            return load_link_graph(link_graph_path, is_pickle=True)
        else:
            raise ValueError(f"{link_graph_path} is not a file.")

    @staticmethod
    def load_from_elastic_search(args) -> Links:
        return


def main():
    parser = create_args_parser()
    args = parser.parse_args()

    links = Loader.load_from_file(args)
    pr = PageRank(links)
    is_success = pr.compute()
    if is_success:
        top_pages = pr.top()
        print(top_pages)


if __name__ == '__main__':
    main()
