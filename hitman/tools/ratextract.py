import numpy as np
import uproot
import re


class DataExtractor():
    def __init__(self, input_files):
        # Validate all files are valid, do not load invalid files
        valid_files = []
        valid_out_keys = []
        for infile in input_files:
            if self.file_isvalid(infile):
                valid_files.append(infile)
                valid_out_keys.append(self.get_valid_out_key(infile))  # This is a temporary hack

        self.out_keys = valid_out_keys
        self.input_files = valid_files

    # Need to define a better way to get the valid outkey if there are several output trees in file
    def get_valid_out_key(self, infile):
        with uproot.open(infile) as file:
            all_out_keys = [out_key for out_key in file.keys() if out_key.startswith('output')]  # filter metas
            all_out_num = np.array([int(out_key[-1]) for out_key in all_out_keys])
            index = np.argmax(all_out_num)
            return all_out_keys[index]

    def file_isvalid(self, infile):
        try:
            with uproot.open(infile) as file:
                if len(file.keys()) > 1:
                    return True
                else:
                    print("Warning: Opening " + infile + " failed")
                    return False

        except:
            print("Warning: Opening " + infile + " failed")
            return False

    def get_norms(self):
        pass
        return hyp_norm, obs_norm

    def get_hitman_train_data(self):
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcPMTNPE', 'mcPMTID', "mcPEFrontEndTime"], library='np')
        maps = uproot.concatenate([infile + ":meta;1" for infile in self.input_files],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')

        n_hit = np.concatenate(obsdata['mcPMTNPE'])
        idx = np.repeat(np.concatenate(obsdata['mcPMTID']), n_hit)
        nhit = np.array([len(hits) for hits in obsdata['mcPEFrontEndTime']], dtype=np.int32)
        charge_obs = np.stack([
            nhit.astype(np.float32),
            nhit.astype(np.float32)
        ], axis=1
        )
        hit_obs = np.stack([
            maps['pmtX'][0][idx].astype(np.float32),
            maps['pmtY'][0][idx].astype(np.float32),
            maps['pmtZ'][0][idx].astype(np.float32),
            np.concatenate(obsdata['mcPEFrontEndTime']).astype(np.float32),
            (np.zeros(len(idx)) + 1).astype(np.float32)
        ]
            , axis=1)

        del obsdata

        hypdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcx', 'mcy', 'mcz', 'mcu', 'mcv', 'mcw', 'mcke'], library='np')
        mcaz = np.mod(np.arctan2(hypdata['mcv'], hypdata['mcu']), 2 * np.pi).astype(np.float32)
        mcze = np.arccos(hypdata['mcw']).astype(np.float32)
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

    def get_hitman_reco_data(self):  # Loads data in old format using python dicts
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcPMTNPE', 'mcPMTID', "mcPEFrontEndTime"], library='np')
        maps = uproot.concatenate([infile + ":meta;1" for infile in self.input_files],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')

        hypdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcx', 'mcy', 'mcz', 'mcu', 'mcv', 'mcw', 'mcke'], library='np')
        mcaz = np.mod(np.arctan2(hypdata['mcv'], hypdata['mcu']), 2 * np.pi).astype(np.float32)
        mcze = mcze = np.arccos(hypdata['mcw']).astype(np.float32)
        mct = np.zeros(len(mcze), np.float32)
        charge_hyp = np.stack([hypdata['mcx'].astype(np.float32),
                               hypdata['mcy'].astype(np.float32),
                               hypdata['mcz'].astype(np.float32),
                               mcze,
                               mcaz,
                               mct.astype(np.float32),
                               hypdata['mcke'].astype(np.float32)
                               ], axis=1)

        events = []

        for i in range(len(mcze)):

            n_hit = obsdata['mcPMTNPE'][i]
            idx = np.repeat(obsdata['mcPMTID'][i], n_hit)

            charge_obs = np.stack([
                np.sum(n_hit),
                np.sum(n_hit)
            ]
            )
            hits = np.stack([
                maps['pmtX'][0][idx].astype(np.float32),
                maps['pmtY'][0][idx].astype(np.float32),
                maps['pmtZ'][0][idx].astype(np.float32),
                obsdata['mcPEFrontEndTime'][i].astype(np.float32),
                (np.zeros(len(idx)) + 1).astype(np.float32)
            ]
                , axis=1)

            event = {
                "hits": hits,
                "total_charge": charge_obs,
                "truth": charge_hyp[i]
            }
            events.append(event)
        return events
