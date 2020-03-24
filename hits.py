import argparse
import elasticsearch
import random
import pathlib
import logging
import urllib.parse
from enum import Enum
from elasticsearch import Elasticsearch
from os import linesep

import numpy as np

import util
from logger import get_logger
from graph import Links

log_file = pathlib.Path.cwd().joinpath("logs").joinpath("hits.log")
log = get_logger(__name__, str(log_file))
log.setLevel(logging.INFO)


class Page(object):
    def __init__(self, _id, url, out_links, in_links):
        self._id = _id
        self._url = url
        self._out_links = out_links
        self._in_links = in_links

    @property
    def id(self):
        return self._id

    @property
    def url(self):
        return self._url

    @property
    def out_links(self):
        return self._out_links

    @property
    def in_links(self):
        return self._in_links

    def __eq__(self, other):
        return isinstance(other, Page) and other.id == self._id

    def __hash__(self):
        return hash(self._id)

    @classmethod
    def deserialize(cls, data):
        source = data["_source"]
        source["_id"] = data["_id"]
        return cls(**source)


def get_page(client, index, _id):
    try:
        doc = client.get(index=index, id=_id, _source_includes=["url", "out_links", "in_links"])
    except elasticsearch.exceptions.NotFoundError as e:
        return None
    else:
        return Page.deserialize(doc)


def create_root_set(client, index, query, size=1000):
    body = {
        "query": {
            "match": {
                "text": {
                    "query": query
                }
            }
        }
    }
    res = client.search(body=body, index=index, size=size, _source_includes=["url", "out_links", "in_links"])
    return res["hits"]["hits"]


def create_base_set(client, index, root_set, required=10000, d=50):
    base_set = set(root_set)
    processed = set([page.id for page in root_set])
    queue = list(root_set)

    def get_pages(links):
        result = []
        ids = [util.hash_id(link) for link in set(links)]
        ids = set(filter(lambda x: x not in processed, ids))
        processed.update(ids)

        if len(ids) > 0:
            res = client.mget(body={"ids": list(ids)}, index=index,
                              _source_includes=["url", "out_links", "in_links"])
            for doc in res["docs"]:
                if doc["found"]:
                    result.append(Page.deserialize(doc))
        return result

    cur_iter = 0
    while len(base_set) <= required:
        cur_iter += 1
        log.info(f"[INFO] Expanding the base set iteration {cur_iter}")

        tmp_queue = []
        # Add all pages that the page points to
        for page in queue:
            pages = get_pages(page.out_links)
            tmp_queue.extend(pages)
            log.info(f"[INFO] Processed {len(pages)} child pages of the page {page.id}")
        child_pages_count = len(tmp_queue)
        log.info(f"[INFO] Added child pages of the current page")

        # Add pages that points to the page
        for page in queue:
            if len(page.in_links) <= d:
                pages = get_pages(page.in_links)
            else:
                pages = get_pages(random.sample(page.in_links, d))
            tmp_queue.extend(pages)
            log.info(f"[INFO] Processed {len(pages)} parent pages of the page {page.id}")
        parent_pages_count = len(tmp_queue) - child_pages_count
        log.info(f"[INFO] Adding a total of {len(tmp_queue)} pages to the base set, "
                 f"num_child_page={child_pages_count}, num_parent_page_count={parent_pages_count}")
        base_set.update(set(tmp_queue))
        queue = random.sample(tmp_queue, len(root_set) // 4)

    log.info(f"[INFO] Find base set size {len(base_set)}")
    return base_set


def pages_to_links_adapter(pages):
    in_links = dict()
    out_links = dict()
    for page in pages:
        url_obj = urllib.parse.urlparse(page.url)
        url_id = page.url if url_obj.scheme == "" else page.url[len(f"{url_obj.scheme}://"):]
        in_links[url_id] = set(page.in_links)
        out_links[url_id] = set(page.out_links)

    return Links(ins=in_links, outs=out_links)


class ScoreType(Enum):
    AUTHORITY = 1
    HUB = 2


class Hits(object):
    def __init__(self, links):
        self._links = links
        self._hub_scores = None
        self._authority_scores = None

    def compute(self, max_iter=50):
        def normalize_vector(v):
            return v / np.linalg.norm(v)

        def check_convergence(old, new, max_diff):
            diff = old - new
            boolean_filters = [value <= max_diff for value in diff]
            print(f"[INFO] Need {boolean_filters.count(False)} more nodes to converge")
            return boolean_filters.count(False) == 0

        nodes_list = list(self._links.ins.keys())
        n = len(self._links.ins)
        id_by_node = {node: _id for _id, node in enumerate(nodes_list)}

        out_links = {
            node: [to_node for to_node in to_nodes if to_node in self._links.ins]
            for node, to_nodes in self._links.outs.items()
        }
        adjacency = np.zeros((n, n), dtype=np.bool_)
        for node in nodes_list:
            node_id = id_by_node[node]
            for to_node in out_links[node]:
                to_node_id = id_by_node[to_node]
                adjacency[node_id, to_node_id] = True
        adjacency_t = np.transpose(adjacency)

        tolerance = (1 / (n * 100))
        hub_scores = np.array([1] * n)
        authority_scores = np.array([1] * n)
        cur_iter = 0
        while True:
            cur_iter += 1
            if cur_iter > max_iter:
                break
            log.info(f"[INFO] Hubs and Authorities Update Iteration {cur_iter}")

            old_authority_scores = authority_scores
            old_hub_scores = hub_scores

            authority_scores = adjacency_t.dot(hub_scores)
            hub_scores = adjacency.dot(authority_scores)
            authority_scores = normalize_vector(authority_scores)
            hub_scores = normalize_vector(hub_scores)

            if (check_convergence(old_authority_scores, authority_scores, tolerance)
                    and check_convergence(old_hub_scores, hub_scores, tolerance)):
                log.info(f"[INFO] Both hubs and authorities scores converged.")
                log.info(f"[INFO] Sum of squared hub scores is "
                         f"{sum(value * value for value in hub_scores)}")
                log.info(f"[INFO] Sum of squared authority scores is "
                         f"{sum(value * value for value in authority_scores)}")
                break

        self._hub_scores = list(zip(nodes_list, hub_scores))
        self._authority_scores = list(zip(nodes_list, authority_scores))

    def top(self, score_type, size=500):
        if score_type is ScoreType.AUTHORITY:
            sorted_nodes = sorted(self._authority_scores, key=lambda t: t[1], reverse=True)
        elif score_type is ScoreType.HUB:
            sorted_nodes = sorted(self._hub_scores, key=lambda t: t[1], reverse=True)
        else:
            raise ValueError(f"{score_type} is not supported.")
        return sorted_nodes[:size]


def get_links(args):
    if args.elastic:
        client = Elasticsearch(args.hostname)
        root_set = set(Page.deserialize(doc)
                       for doc in create_root_set(client, args.index, args.query, size=1000))
        base_set = create_base_set(client, args.index, root_set)
        return pages_to_links_adapter(base_set)
    else:
        pickle_path = pathlib.Path(args.path)
        return pages_to_links_adapter(util.load_pickle(pickle_path))


def remove_intrinsic_links(links):
    def hostname(i):
        # Notice adding // in front of the url to make urlparse work correctly.
        url_obj = urllib.parse.urlparse(f"//{i}")
        return url_obj.hostname

    ins = dict()
    outs = dict()
    for url in links.ins.keys():
        url_hostname = hostname(url)
        ins[url] = list(filter(lambda x: hostname(x) != url_hostname, links.ins[url]))
        outs[url] = list(filter(lambda x: hostname(x) != url_hostname, links.outs[url]))
    return Links(ins=ins, outs=outs)


def file_writer(scored_pages, path):
    with path.open("w") as fp:
        for _id, score in scored_pages:
            fp.write(f"{_id}\t{score}{linesep}")


def create_args_parser():
    parser = argparse.ArgumentParser(description="HITS Computation")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--elastic", action="store_true",
                              help="specify the source is from ElasticSearch")
    source_group.add_argument("--pickle", action="store_true",
                              help="specify the source is from a pickle file")
    parser.add_argument("-p", "--path", action="store",
                        help="the path of the pickle file")
    parser.add_argument("-hst", "--hostname", action="store",
                        help="hostname of the ElasticSearch instance")
    parser.add_argument("-i", "--index", action="store",
                        help="the name of the index")
    parser.add_argument("-q", "--query", action="store",
                        help="a query to match a root set of documents")
    return parser


def main():
    parser = create_args_parser()
    args = parser.parse_args()
    links = get_links(args)
    intrinsic_links_removed = remove_intrinsic_links(links)
    hits = Hits(intrinsic_links_removed)
    hits.compute()
    top_authorities = hits.top(ScoreType.AUTHORITY)
    top_hubs = hits.top(ScoreType.HUB)

    results_folder = pathlib.Path.cwd().joinpath("data").joinpath("results")
    authority_scores_result_file = results_folder.joinpath("authority_scores.txt")
    hub_scores_result_file = results_folder.joinpath("hub_scores.txt")

    file_writer(top_authorities, authority_scores_result_file)
    file_writer(top_hubs, hub_scores_result_file)


if __name__ == '__main__':
    main()
