from elasticsearch import Elasticsearch
import elasticsearch
import random
import pathlib
import logging

from logger import get_logger
import util

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


class Hits(object):
    def __init__(self, base_set):
        self._base_set = base_set


def main():
    client = Elasticsearch("http://localhost:9200")
    index = "hurricane_v2"
    query = "recent major hurricanes"
    base_set_pickle_path = pathlib.Path.cwd().joinpath("results").joinpath("base_set_pickle")

    root_set = set(Page.deserialize(doc)
                   for doc in create_root_set(client, index, query, size=1000))
    base_set = create_base_set(client, index, root_set)
    util.write_pickle(base_set_pickle_path, base_set)


if __name__ == '__main__':
    main()