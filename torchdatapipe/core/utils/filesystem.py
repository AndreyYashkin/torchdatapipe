import os
import mimetypes
import re
from glob import iglob


def find_mimetype_files(root_dir, mimetype, recursive):
    extensions = []
    for ext, mtype in mimetypes.types_map.items():
        if re.match(mimetype, mtype):
            extensions.append(ext)

    assert len(extensions)
    pattern = re.compile(f".*({'|'.join(extensions)})", flags=re.IGNORECASE)

    all_files = iglob("**", root_dir=root_dir, recursive=recursive)
    files = []
    for path in all_files:
        full_path = os.path.join(root_dir, path)
        if not os.path.isfile(full_path):
            continue
        if pattern.fullmatch(path):
            files.append(path)
    return files
