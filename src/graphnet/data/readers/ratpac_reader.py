"""Modules for reading data files from LiquidO."""

import os
from glob import glob
from typing import List, Union, Dict, Any

import numpy as np
import uproot

from graphnet.data.extractors.ratpac import MCHitExtractor, MCTruthExtractor
from .graphnet_file_reader import GraphNeTFileReader


class NtupleReader(GraphNeTFileReader):
    """A class for reading ntuple ROOT files from ratpac-two."""

    _accepted_file_extensions = [".root"]
    _accepted_extractors = [MCHitExtractor, MCTruthExtractor]

    def __call__(self, file_path: str) -> List[Dict[str, Dict[str, Any]]]:
        outputs = []

        with uproot.open(file_path) as file:
            out_key = self.get_valid_out_key(file)
            obsdata = file[out_key].arrays(
                filter_name=[
                    'mcPMTNPE', 'mcPMTID', 'mcPEFrontEndTime',
                    'mcx', 'mcy', 'mcz', 'mcu', 'mcv', 'mcw', 'mct', 'mcke', 'mcpdg'
                ],
                library='np'
            )
            maps = file["meta;1"].arrays(
                filter_name=['pmtX', 'pmtY', 'pmtZ', 'pmtU', 'pmtV', 'pmtW'],
                library='np'
            )

            n_events = len(obsdata['mcx'])  # Number of events

            for i in range(n_events):
                event_data = {key: obsdata[key][i] for key in obsdata.keys()}
                event_outputs = {}

                for extractor in self._extractors:
                    if isinstance(extractor, MCHitExtractor):
                        data = extractor(event_data, maps)
                    else:
                        data = extractor(event_data)
                    if data is not None:
                        event_outputs[extractor._extractor_name] = data

                if event_outputs:
                    outputs.append(event_outputs)

        return outputs

    def get_valid_out_key(self, file) -> str:
        """Determine the valid output key by selecting the one with the highest numeric suffix."""
        out_keys = [key for key in file.keys() if key.startswith('output')]
        if not out_keys:
            raise ValueError(f"No valid output keys found in file.")
        out_num = np.array([int(key[-1]) for key in out_keys])
        return out_keys[np.argmax(out_num)]

    def find_files(self, path: Union[str, List[str]]) -> List[str]:
        """Search folder(s) for ROOT files."""
        files = []
        if isinstance(path, str):
            path = [path]
        for p in path:
            files.extend(glob(os.path.join(p, "*.root")))
        return files
