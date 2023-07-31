import numpy as np
import uproot
import re


class DataExtractor():
    def __init__(self, input_files):
        # Validate all files are valid, do not load invalid files
        valid_files = []
        valid_out_keys = []
        valid_meta_keys = []

        for infile in input_files:
            if self.file_isvalid(infile):
                valid_files.append(infile)
                valid_out_keys.append(self.get_valid_out_key(infile))  # This is a temporary hack
                valid_meta_keys.append(self.get_valid_meta_key(infile))  # This is a temporary hack

        self.out_keys = valid_out_keys
        self.meta_keys = valid_meta_keys
        self.input_files = valid_files

    # Need to define a better way to get the valid outkey if there are several output trees in file
    def get_valid_out_key(self, infile):
        with uproot.open(infile) as file:
            all_out_keys = [out_key for out_key in file.keys() if out_key.startswith('output')]  # filter metas
            all_out_num = np.array([int(out_key[-1]) for out_key in all_out_keys])
            index = np.argmax(all_out_num)
            return all_out_keys[index]

    def get_valid_meta_key(self, infile):
        with uproot.open(infile) as file:
            all_out_keys = [out_key for out_key in file.keys() if out_key.startswith('meta')]  # filter metas
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
            filter_name=['mcPEIndex', 'mcPMTID', "mcPETime", "hitPMTCharge"], library='np')
        maps = uproot.concatenate([infile + ":meta;1" for infile in self.input_files],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')

        pe_idx = np.concatenate(obsdata['mcPEIndex'])
        idx = np.where(pe_idx == 0)  # Get PMT Hits
        n_hit = np.diff(idx, append=len(pe_idx))[0]  # Count n_pe per sensor
        idx = np.repeat(np.concatenate(obsdata['mcPMTID']), n_hit)
        nhit = np.array([len(hits) for hits in obsdata['mcPETime']], dtype=np.int32)
        charge_obs = np.stack([
            np.array([np.sum(charge) for charge in obsdata['hitPMTCharge']], dtype=np.float32),
            nhit.astype(np.float32)
        ], axis=1
        )
        hit_obs = np.stack([
            maps['pmtX'][0][idx].astype(np.float32),
            maps['pmtY'][0][idx].astype(np.float32),
            maps['pmtZ'][0][idx].astype(np.float32),
            np.concatenate(obsdata['mcPETime']).astype(np.float32),
            (np.zeros(len(pe_idx)) + 1).astype(np.float32)
        ]
            , axis=1)

        del obsdata

        charge_hyp = self.get_truth()

        hit_hyp = np.repeat(charge_hyp, nhit, axis=0)
        return charge_obs, hit_obs, charge_hyp, hit_hyp

    def get_truth(self):
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
        return charge_hyp
    def get_source_truth(self):
        # Ignores z rotation...
        sourcedata = uproot.concatenate(
            [self.input_files[i] + ":" + self.meta_keys[i] for i in range(len(self.input_files))],
            filter_name=['source_pos_x', 'source_pos_y', 'source_pos_z', 'source_rot_x', 'source_rot_y'], library='np')

        hypdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcke', 'evid'], library='np')

        #Get number of triggered events in each file
        trg_events = -np.diff(np.append(hypdata['evid'], 0))
        trg_events = trg_events[trg_events >= 0] + 1


        # use the transpose of the euler rotation matrix to get the source direction assuming the beta window
        # is (0,0,-1) in the coordinate system of the source

        src_mcu = np.sin(sourcedata['source_rot_y'] * np.pi/180)
        src_mcv = -np.cos(sourcedata['source_rot_y'] * np.pi/180) * np.sin(sourcedata['source_rot_x'] * np.pi/180)
        src_mcw = -np.cos(sourcedata['source_rot_y'] * np.pi/180) * np.cos(sourcedata['source_rot_x'] * np.pi/180)
        print(src_mcu, src_mcv, src_mcw)
        print(sourcedata['source_rot_x'], sourcedata['source_rot_y'])
        #repeat these values for each event
        src_mcu = np.repeat(src_mcu, trg_events)
        src_mcv = np.repeat(src_mcv, trg_events)
        src_mcw = np.repeat(src_mcw, trg_events)
        src_mcx = np.repeat(sourcedata['source_pos_x'], trg_events)
        src_mcy = np.repeat(sourcedata['source_pos_y'], trg_events)
        src_mcz = np.repeat(sourcedata['source_pos_z'], trg_events)




        #add the time to the source data
        mct = np.zeros(len(hypdata['mcke']), np.float32)

        mcaz = np.mod(np.arctan2(src_mcv, src_mcu), 2 * np.pi).astype(np.float32)
        mcze = np.arccos(src_mcw).astype(np.float32)

        print("Source data shape: ", src_mcx.shape)
        print("Hyp data shape: ", hypdata['mcke'].shape)

        charge_hyp = np.stack([src_mcx.astype(np.float32),
                               src_mcy.astype(np.float32),
                               src_mcz.astype(np.float32),
                               mcze.astype(np.float32),
                               mcaz.astype(np.float32),
                               mct.astype(np.float32),
                               hypdata['mcke'].astype(np.float32)
                               ], axis=1)
        return charge_hyp

    def get_hitman_reco_data(self):  # Loads data in old format using python dicts
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcPEIndex', 'mcPMTID', "mcPETime", "hitPMTCharge"], library='np')
        maps = uproot.concatenate([infile + ":meta;1" for infile in self.input_files],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')

        charge_hyp = self.get_truth()

        events = []

        for i in range(len(charge_hyp)):
            pe_idx = obsdata['mcPEIndex'][i]
            idx = np.where(pe_idx == 0)[0]  # Get PMT Hits
            n_hit = np.diff(idx, append=len(pe_idx))  # Count n_pe per sensor
            idx = np.repeat(obsdata['mcPMTID'][i], n_hit)

            charge_obs = np.stack([
                np.sum(obsdata['hitPMTCharge'][i]),
                np.sum(n_hit)
            ]
            )
            hits = np.stack([
                maps['pmtX'][0][idx].astype(np.float32),
                maps['pmtY'][0][idx].astype(np.float32),
                maps['pmtZ'][0][idx].astype(np.float32),
                obsdata['mcPETime'][i].astype(np.float32),
                (np.zeros(len(pe_idx)) + 1).astype(np.float32)
            ]
                , axis=1)

            event = {
                "hits": hits,
                "total_charge": charge_obs,
                "truth": charge_hyp[i]
            }
            if len(obsdata['hitPMTCharge'][i]) > 3:
                events.append(event)
        return events
