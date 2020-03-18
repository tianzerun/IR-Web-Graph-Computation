import pickle
import pathlib
import argparse
from collections import namedtuple

Links = namedtuple("Links", "ins, outs")


def create_args_parser():
    parser = argparse.ArgumentParser(description='Link graph generation')
    parser.add_argument("link_graph", metavar="link-graph", action="store",
                        help="path of the link-graph file (either text or pickle)")
    parser.add_argument("--is-pickle", action="store_true",
                        help="specify the file is a pickled object")
    return parser


def load_link_graph(path, is_pickle=False):
    if is_pickle:
        with path.open("rb") as fp:
            return pickle.load(fp)

    out_links = dict()
    in_links = dict()
    counter = 0
    with path.open("r") as fp:
        for line in fp:
            urls = line.rstrip().split("\t")
            out_links[urls[0]] = set(urls[1:])
            in_links[urls[0]] = set()
            counter += 1

    for from_url, to_urls in out_links.items():
        for url in to_urls:
            if url in in_links:
                in_links[url].add(from_url)

    return Links(ins=in_links, outs=out_links)


def main():
    parser = create_args_parser()
    args = parser.parse_args()
    link_graph_path = pathlib.Path(args.link_graph)
    if link_graph_path.is_file():
        links = load_link_graph(link_graph_path, args.is_pickle)

        if not args.is_pickle:
            with link_graph_path.parent.joinpath(f"{link_graph_path.stem}_pickle").open("wb") as fp:
                pickle.dump(links, fp)
    else:
        raise ValueError(f"{link_graph_path} is not a file.")


if __name__ == '__main__':
    main()
