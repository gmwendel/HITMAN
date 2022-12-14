import numpy as np
import uproot


class DataExtractor():
    def __init__(self, input_files):
        # Validate all files are valid, do not load invalid files
        valid_files = []
        for infile in input_files:
            if self.file_isvalid(infile):
                valid_files.append(infile)
        self.input_files = valid_files

    def file_isvalid(self, infile):
        try:
            with uproot.open(infile + ":output;1") as file:
                pass
            return True
        except:
            print("Warning: Opening " + infile + " failed")
            return False

    def get_norms(self):
        pass
        return hyp_norm, obs_norm

    def get_hitman_train_data(self):
        obsdata = uproot.concatenate([infile + ":output;1" for infile in self.input_files],
                                     filter_name=["hitPMTID", "hitPMTTime", "hitPMTCharge"], library='np')
        maps = uproot.concatenate([infile + ":meta;1" for infile in self.input_files],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')

        idx = np.concatenate(obsdata['hitPMTID'])
        nhit = np.array([len(charge) for charge in obsdata['hitPMTCharge']], dtype=np.int32)
        charge_obs = np.stack([
            np.array([np.sum(charge) for charge in obsdata['hitPMTCharge']], dtype=np.float32),
            nhit.astype(np.float32)
        ],axis=1
        )
        hit_obs = np.stack([
            maps['pmtX'][0][idx].astype(np.float32),
            maps['pmtY'][0][idx].astype(np.float32),
            maps['pmtZ'][0][idx].astype(np.float32),
            np.concatenate(obsdata['hitPMTTime']).astype(np.float32),
            np.concatenate(obsdata['hitPMTCharge']).astype(np.float32)
        ]
            , axis=1)

        del obsdata

        hypdata = uproot.concatenate([infile + ":output;1" for infile in self.input_files],
                                     filter_name=['mcx', 'mcy', 'mcz', 'mcu', 'mcv', 'mcw', 'mcke'], library='np')
        mcaz = np.mod(np.arctan2(hypdata['mcv'], hypdata['mcu']), 2 * np.pi).astype(np.float32)
        mcze = np.arccos(hypdata['mcw'] / np.linalg.norm([hypdata['mcu'], hypdata['mcv'], hypdata['mcw']], 2)).astype(
            np.float32)
        mct = np.zeros(len(mcze), np.float32)
        charge_hyp = np.stack([hypdata['mcx'].astype(np.float32),
                               hypdata['mcy'].astype(np.float32),
                               hypdata['mcz'].astype(np.float32),
                               mcze,
                               mcaz,
                               mct.astype(np.float32),
                               hypdata['mcke'].astype(np.float32)
                               ], axis=1)

        hit_hyp = np.repeat(charge_hyp, nhit, axis=0)
        return charge_obs, hit_obs, charge_hyp, hit_hyp