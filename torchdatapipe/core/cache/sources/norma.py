import os
import json
from glob import glob
from .source import Source


default_dataset_attributes = [
    "camera.intrinsics",
    "camera.rt",
    "resolution.height",
    "resolution.width",
]
default_scene_attributes = ["scene", "scene_depth"]
default_object_attributes = [
    "id",
    "class_id",
    "rotation",
    "translation",
    "visible_mask",
    "full_mask",
]


def _load_attribute(d, attr):
    keys = attr.split(".")
    for key in keys:
        if key in d:
            d = d[key]
        else:
            return None
    return d


def _load_attributes(d, attrs):
    values = {}
    for attr in attrs:
        values[attr] = _load_attribute(d, attr)
    return values


class NormaSource(Source):
    def __init__(
        self,
        root,
        dataset_attributes=default_dataset_attributes,
        scene_attributes=default_scene_attributes,
        object_attributes=default_object_attributes,
    ):
        self.__root = root
        self.dataset_attributes = dataset_attributes
        self.scene_attributes = scene_attributes
        self.object_attributes = object_attributes

    @property
    def root(self):
        return self.__root

    @property
    def version(self):
        return "0.0.0"

    @property
    def params(self):
        return dict(
            root=self.root,
            dataset_attributes=self.dataset_attributes,
            scene_attributes=self.scene_attributes,
            object_attributes=self.object_attributes,
        )

    def start_caching(self):
        self.dataset_name = os.path.basename(self.root)
        dataset_annotation_path = os.path.join(self.root, self.dataset_name + ".json")
        with open(dataset_annotation_path) as f:
            dataset_annotation = json.load(f)
        self.ds_attrs_values = _load_attributes(dataset_annotation, self.dataset_attributes)
        scenes_dir = os.path.join(self.root, "scenes")
        self.scene_files = glob("*/*.json", root_dir=scenes_dir)

    def __len__(self):
        return len(self.scene_files)

    def __getitem__(self, idx):
        scenes_dir = os.path.join(self.root, "scenes")
        scene_path = os.path.join(scenes_dir, self.scene_files[idx])

        name = os.path.basename(scene_path)[:-5]
        scene_dir = os.path.dirname(scene_path)

        with open(scene_path) as f:
            scene = json.load(f)

        scene_attrs = _load_attributes(scene, self.scene_attributes)

        objects = []
        for obj in scene["objects"]:
            obj_attrs = _load_attributes(obj, self.object_attributes)
            objects.append(obj_attrs)

        return dict(
            name=name,
            scene_dir=scene_dir,
            dataset_attrs=self.ds_attrs_values,
            scene_attrs=scene_attrs,
            objects=objects,
        )

    def finish_caching(self):
        self.ds_attrs_values = None
        self.scene_files = None
