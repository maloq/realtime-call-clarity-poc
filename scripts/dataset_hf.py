# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DAPS Dataset"""

import glob
import os

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{mysore2014can,
  title={Can we automatically transform speech recorded on common consumer devices in real-world environments into professional production quality speech?—a dataset, insights, and challenges},
  author={Mysore, Gautham J},
  journal={IEEE Signal Processing Letters},
  volume={22},
  number={8},
  pages={1006--1010},
  year={2014},
  publisher={IEEE}
}
"""

# You can copy an official description
_DESCRIPTION = """\
The DAPS (Device and Produced Speech) dataset is a collection of aligned versions of professionally produced studio speech recordings and recordings of the same speech on common consumer devices (tablet and smartphone) in real-world environments. It has 15 versions of audio (3 professional versions and 12 consumer device/real-world environment combinations). Each version consists of about 4 1/2 hours of data (about 14 minutes from each of 20 speakers).
"""

_HOMEPAGE = "https://ccrma.stanford.edu/~gautham/Site/daps.html"

_LICENSE = "Creative Commons Attribution Non Commercial 4.0 International"

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = "https://zenodo.org/record/4660670/files/daps.tar.gz"


class DapsDataset(datasets.GeneratorBasedBuilder):
    """The DAPS (Device and Produced Speech) dataset is a collection of aligned versions of professionally produced studio speech recordings and recordings of the same speech on common consumer devices (tablet and smartphone) in real-world environments."""

    VERSION = datasets.Version("2.12.0")

    DEFAULT_CONFIG_NAME = "aligned_examples"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features(
            {
                "recording_environment": datasets.Value("string"),
                "speaker_id": datasets.Value("string"),
                "script_id": datasets.Value("string"),
                "clean_path": datasets.Value("string"),
                "produced_path": datasets.Value("string"),
                "device_path": datasets.Value("string"),
                "clean_audio": datasets.Audio(sampling_rate=44_100),
                "produced_audio": datasets.Audio(sampling_rate=44_100),
                "device_audio": datasets.Audio(sampling_rate=44_100),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS
        data_dir = dl_manager.download_and_extract(urls)
        daps_dir = os.path.join(data_dir, "daps")
        if os.path.isdir(daps_dir):
            data_dir = daps_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                },
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath):
        gt = ["clean", "produced"]
        environments = [
            "ipad_balcony1",
            "ipad_livingroom1",
            "ipadflat_office1",
            "ipad_bedroom1",
            "ipad_office1",
            "iphone_balcony1",
            "ipad_confroom1",
            "ipad_office2",
            "iphone_bedroom1",
            "ipad_confroom2",
            "ipadflat_confroom1",
            "iphone_livingroom1",
        ]
        # example path: daps/iphone_bedroom1/m8_script5_iphone_bedroom1.wav
        for env in environments:
            for device_path in glob.glob(os.path.join(filepath, env) + "/*.wav"):
                speaker_id = os.path.basename(device_path).split("_")[-4]
                script_id = os.path.basename(device_path).split("_")[-3]
                clean_path = device_path.replace(env, "clean")
                produced_path = device_path.replace(env, "produced")
                with open(clean_path, "rb") as f:
                    clean_audio = {"path": clean_path, "bytes": f.read()}
                with open(produced_path, "rb") as f:
                    produced_audio = {"path": produced_path, "bytes": f.read()}
                with open(device_path, "rb") as f:
                    device_audio = {"path": device_path, "bytes": f.read()}
                yield f"{speaker_id}_{script_id}_{env}", {
                    "recording_environment": env,
                    "speaker_id": speaker_id,
                    "script_id": script_id,
                    "clean_path": clean_path,
                    "produced_path": produced_path,
                    "device_path": device_path,
                    "clean_audio": clean_audio,
                    "produced_audio": produced_audio,
                    "device_audio": device_audio,
                }
