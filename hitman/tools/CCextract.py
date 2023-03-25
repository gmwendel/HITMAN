import numpy as np
import uproot

import numpy as np
import uproot


class DataExtractor():
    def __init__(self, input_files):
        '''
        :param input_files: list of input files
        Loads a list of input files and extracts the data from the output trees in a root file in various ways
        '''

        if isinstance(input_files, str):
            input_files = [input_files]

        # Validate all files are valid, do not load invalid files
        valid_files = []
        valid_out_keys = []
        valid_init_mc_keys = []
        for infile in input_files:
            if self.file_isvalid(infile):
                valid_files.append(infile)
                valid_out_keys.append(self.get_valid_out_key(infile))  # This is a temporary hack
                valid_init_mc_keys.append(self.get_valid_init_mc_key(infile))

        self.out_keys = valid_out_keys
        self.init_mc_keys = valid_init_mc_keys
        self.input_files = valid_files

    # Need to define a better way to get the valid outkey if there are several output trees in file
    def get_valid_out_key(self, infile):
        with uproot.open(infile) as file:
            all_out_keys = [out_key for out_key in file.keys() if out_key.startswith('op_hits')]  # filter metas
            all_out_num = np.array([int(out_key[-1]) for out_key in all_out_keys])
            index = np.argmax(all_out_num)
            return all_out_keys[index]

    def get_valid_init_mc_key(self, infile):
        with uproot.open(infile) as file:
            try:
                all_out_keys = [out_key for out_key in file.keys() if out_key.startswith('init_mc')]  # filter metas
                all_out_num = np.array([int(out_key[-1]) for out_key in all_out_keys])
                index = np.argmax(all_out_num)
                return all_out_keys[index]
            except:
                print('no init_mc tree found')
                return None

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

    def get_truth_data(self):
        truthdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.init_mc_keys[i] for i in range(len(self.input_files))],
            filter_name=["azimuthalAngle", "polarAngle", "initialEnergy"], library='np')

        az = np.mod(truthdata['azimuthalAngle'], 2 * np.pi).astype(np.float32)

        ze = truthdata['polarAngle'].astype(np.float32)
        E = np.array([time[0] for time in truthdata['i_time']])
        energy = np.array([energy[0] for energy in truthdata['i_E']])

        hyp = np.stack([
            ze,
            az,
            energy
        ], axis=1)

        return hyp

    def get_hitman_train_data(self):
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['channelID', 'timestamp'], library='np')

        nhit = np.array([len(hits) for hits in obsdata['channelID']], dtype=np.int32)
        charge_obs = np.stack([
            nhit.astype(np.float32)
        ], axis=1)

        hit_obs = np.stack([
            np.concatenate(obsdata['channelID']).astype(np.float32),
            np.concatenate(obsdata['timestamp']).astype(np.float32)
        ]
            , axis=1)

        del obsdata

        charge_hyp = self.get_truth_data()
        hit_hyp = np.repeat(charge_hyp, nhit, axis=0)

        return charge_obs, hit_obs, charge_hyp, hit_hyp

    def get_hitman_reco_data(self):
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['h_primary_id', 'h_time'], library='np')

        nhit = np.array([len(hits) for hits in obsdata['h_primary_id']], dtype=np.int32)
        charge_obs = np.stack([
            nhit.astype(np.float32)
        ], axis=1)

        hit_obs = np.stack([
            np.concatenate(obsdata['h_primary_id']).astype(np.float32),
            np.concatenate(obsdata['h_time']).astype(np.float32)
        ]
            , axis=1)

        hit_obs[:, 1] = hit_obs[:, 1] + np.random.exponential(10, len(hit_obs[:, 1]))  # Add random time decay

        del obsdata

        charge_hyp = self.get_init_truth_data()
        hit_hyp = np.repeat(charge_hyp, nhit, axis=0)

        return charge_obs, hit_obs, charge_hyp, hit_hyp
