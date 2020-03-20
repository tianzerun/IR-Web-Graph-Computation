import pickle
import pathlib
import argparse
from collections import namedtuple

Links = namedtuple("Links", "ins, outs")


def create_args_parser():
    parser = argparse.ArgumentParser(description='Link graph generation')
    parser.add_argument("link_graph", metavar="link-graph", action="store",
                        help="path of the link-graph file (either text or pickle)")
    file_type_group = parser.add_mutually_exclusive_group(required=True)
    file_type_group.add_argument("--in-links", action="store_true")
    file_type_group.add_argument("--out-links", action="store_true")
    parser.add_argument("--is-pickle", action="store_true",
                        help="specify the file is a pickled object")
    return parser


def load_link_graph(path, is_out_links, is_pickle=False):
    if is_pickle:
        with path.open("rb") as fp:
            return pickle.load(fp)

    if is_out_links:
        out_links = LinkGraphHandler.load_links_graph_file(path)
        in_links = LinkGraphHandler.build_in_links_by_out_links(out_links)
    else:
        in_links = LinkGraphHandler.load_links_graph_file(path)
        out_links = LinkGraphHandler.build_out_links_by_in_links(in_links)

    return Links(ins=in_links, outs=out_links)


class LinkGraphHandler(object):
    @staticmethod
    def load_links_graph_file(path):
        mapping = dict()
        with path.open("r") as fp:
            for line in fp:
                urls = line.rstrip().split()
                mapping[urls[0]] = set(urls[1:])
        return mapping

    @staticmethod
    def _reversed_relationships(mapping):
        counter_part = {node: set() for node in mapping.keys()}
        for key, values in mapping.items():
            for v in values:
                if v in counter_part:
                    counter_part[v].add(key)
        return counter_part

    @staticmethod
    def build_in_links_by_out_links(mapping):
        return LinkGraphHandler._reversed_relationships(mapping)

    @staticmethod
    def build_out_links_by_in_links(mapping):
        return LinkGraphHandler._reversed_relationships(mapping)


def main():
    parser = create_args_parser()
    args = parser.parse_args()
    link_graph_path = pathlib.Path(args.link_graph)
    if link_graph_path.is_file():
        links = load_link_graph(link_graph_path, args.out_links, args.is_pickle)

        if not args.is_pickle:
            with link_graph_path.parent.joinpath(f"{link_graph_path.stem}_pickle").open("wb") as fp:
                pickle.dump(links, fp)
    else:
        raise ValueError(f"{link_graph_path} is not a file.")


if __name__ == '__main__':
    main()
