import os
import json
from joblib import Parallel, delayed
from tqdm import tqdm


__VERSION = "0.0.1"


def writer_root_2_cache_json(root):
    return os.path.abspath(root) + "_cache.json"


def __cache_item(source, preprocessor, writer, idx):
    item = source[idx]
    if not item:
        return None
    items = preprocessor(item)
    for i, item in enumerate(items):
        if item is not None:
            writer.write(item, idx, i)


def cache(source, preprocessor, writer, n_jobs):
    writer.clear()
    for element in [source, preprocessor, writer]:
        element.start_caching()
    runner = Parallel(n_jobs=n_jobs, backend="threading")
    runner(
        delayed(__cache_item)(source, preprocessor, writer, idx) for idx in tqdm(range(len(source)))
    )
    for element in [source, preprocessor, writer]:
        element.finish_caching()


def get_cache_description(source, preprocessor, writer):
    source_description = source.cache_description()
    preprocessor_description = preprocessor.cache_description()
    writer_description = writer.cache_description()

    return dict(
        version=__VERSION,
        source=source_description,
        preprocessor=preprocessor_description,
        writer=writer_description,
    )


def write_cache_description(source, preprocessor, writer, file_fn=writer_root_2_cache_json):
    description = get_cache_description(source, preprocessor, writer)
    # В словаре int -> val меняем на str -> val
    json_object = json.dumps(description, indent=4)
    json_path = file_fn(writer.root)
    with open(json_path, "w") as outfile:
        outfile.write(json_object)


def delete_cache_description(writer, file_fn=writer_root_2_cache_json):
    json_path = file_fn(writer.root)
    if os.path.isfile(json_path):
        os.remove(json_path)


def check_cache_description(source, preprocessor, writer, file_fn=writer_root_2_cache_json):
    if os.environ.get("TORCHDATAPIPE_FORCE_CACHE"):
        return False
    json_path = file_fn(writer.root)
    if not os.path.isfile(json_path):
        return False
    with open(json_path) as f:
        old_description = json.load(f)
    new_description = get_cache_description(source, preprocessor, writer)
    new_description = json.loads(json.dumps(new_description))
    return new_description == old_description
