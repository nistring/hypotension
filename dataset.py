import numpy as np
import pandas as pd
import vitaldb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils import preprocessing
from tqdm import tqdm

np.random.seed(42)


class VitalDataset(Dataset):
    def __init__(
        self,
        cfg,
        data_split: tuple,
        trks_csv="https://api.vitaldb.net/trks",
        cases_csv="https://api.vitaldb.net/cases",
        tname="Solar8000/ART_MBP",
    ):
        self.min_ahead = cfg.MINUTES_AHEAD
        self.sampling_rate = cfg.SAMPLING_RATE
        self.in_horizon = int(cfg.IN_HORIZON // self.sampling_rate)
        self.out_horizon = int(cfg.OUT_HORIZON // self.sampling_rate)

        self.tname = tname
        df_trks = pd.read_csv(trks_csv)
        df_cases = pd.read_csv(cases_csv)
        caseids = list(
            set(df_trks[df_trks["tname"] == self.tname]["caseid"])
            & set(df_cases[df_cases["age"] > 18]["caseid"])
            & set(df_cases[~df_cases["opname"].str.contains("transplant")]["caseid"])
        )
        caseids = caseids[:min(cfg.MAX_CASES, len(caseids))]
        caseids = caseids[
            int(len(caseids) * data_split[0]) : min(
                int(len(caseids) * data_split[1]), len(caseids) - 1
            )
        ]

        self.caseids = []
        self.page_len = []
        np.random.shuffle(caseids)
        print("Loading data..")
        for caseid in tqdm(caseids):
            mbps = vitaldb.load_case(caseid, [self.tname], 1 / self.sampling_rate)
            x, y = preprocessing(mbps, self.in_horizon, self.out_horizon)
            if len(y) > 0:
                self.caseids.append(caseid)
                self.page_len.append(len(y))

        print("Total {} cases found".format(len(self.caseids)))
        print("Total {} samples found".format(len(self)))
        self._reset_page()
        self._load_vitaldb()

    def __len__(self):
        return sum(self.page_len)

    def __getitem__(self, index):
        if self.page_len[self.page] <= index - self.page_idx:
            self._increment_page()
            self._load_vitaldb()
        distance = index - self.page_idx
        if index == len(self) - 1:
            self._reset_page()
        
        return self.x[distance], self.y[distance]

    def _reset_page(self):
        self.page = 0
        self.page_idx = 0

    def _increment_page(self):
        self.page_idx += self.page_len[self.page]
        self.page += 1

    def _load_vitaldb(self):
        self.x, self.y = preprocessing(
            vitaldb.load_case(
                self.caseids[self.page], [self.tname], 1 / self.sampling_rate
            ),
            self.in_horizon,
            self.out_horizon,
        )
