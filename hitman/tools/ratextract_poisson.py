import numpy as np
import uproot
from hitman.tools.ratextract import DataExtractor

class PoissonDataExtractor(DataExtractor):
    def __init__(self, input_files):
        super().__init__(input_files)

    def get_poisson_train_data(self):
        # First, grab the standard hit data
        _, hit_obs, charge_hyp, hit_hyp = super().get_hitman_train_data()
        
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcPMTNPE', 'mcPMTID'], library='np')
        maps = uproot.concatenate([self.input_files[0] + ":meta;1"],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')
        
        pmt_positions = np.stack([
            maps['pmtX'][0].astype(np.float32),
            maps['pmtY'][0].astype(np.float32),
            maps['pmtZ'][0].astype(np.float32)
        ], axis=1)
        
        N_sensors = len(pmt_positions)
        N_events = len(charge_hyp)
        
        event_lengths = np.array([len(x) for x in obsdata['mcPMTID']], dtype=np.int32)
        event_indices = np.repeat(np.arange(N_events), event_lengths)
        flat_pmt_ids = np.concatenate(obsdata['mcPMTID'])
        flat_npes = np.concatenate(obsdata['mcPMTNPE'])
        
        charges = np.zeros((N_events, N_sensors), dtype=np.float32)
        np.add.at(charges, (event_indices, flat_pmt_ids), flat_npes)
            
        return charges, charge_hyp, pmt_positions, hit_obs, hit_hyp

    def get_poisson_only_train_data(self):
        # Optimized loading: exclusively load necessary arrays, entirely skipping the massive mcPEFrontEndTime
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcPMTNPE', 'mcPMTID'], library='np')
        maps = uproot.concatenate([self.input_files[0] + ":meta;1"],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')
        hypdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcke', 'scintmod_scat_len', 'scintmod_abs_len'], library='np')
        
        pmt_positions = np.stack([
            maps['pmtX'][0].astype(np.float32),
            maps['pmtY'][0].astype(np.float32),
            maps['pmtZ'][0].astype(np.float32)
        ], axis=1)
        
        charge_hyp = np.stack([hypdata['mcke'].astype(np.float32),
                               hypdata['scintmod_scat_len'].astype(np.float32),
                               hypdata['scintmod_abs_len'].astype(np.float32)
                               ], axis=1)

        N_sensors = len(pmt_positions)
        N_events = len(charge_hyp)
        
        # Vectorized charge accumulation (bypassing slow Python loops over N_events)
        event_lengths = np.array([len(x) for x in obsdata['mcPMTID']], dtype=np.int32)
        event_indices = np.repeat(np.arange(N_events), event_lengths)
        flat_pmt_ids = np.concatenate(obsdata['mcPMTID'])
        flat_npes = np.concatenate(obsdata['mcPMTNPE'])
        
        charges = np.zeros((N_events, N_sensors), dtype=np.float32)
        np.add.at(charges, (event_indices, flat_pmt_ids), flat_npes)
            
        return charges, charge_hyp, pmt_positions

    def get_poisson_reco_data(self):
        events_nre = super().get_hitman_reco_data()
        
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcPMTNPE', 'mcPMTID'], library='np')
        maps = uproot.concatenate([self.input_files[0] + ":meta;1"],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')
        
        pmt_positions = np.stack([
            maps['pmtX'][0].astype(np.float32),
            maps['pmtY'][0].astype(np.float32),
            maps['pmtZ'][0].astype(np.float32)
        ], axis=1)
        
        N_sensors = len(pmt_positions)
        N_events = len(events_nre)
        events = []
        
        for i in range(N_events):
            charges = np.zeros(N_sensors, dtype=np.float32)
            pmt_ids = obsdata['mcPMTID'][i]
            pmt_npes = obsdata['mcPMTNPE'][i]
            np.add.at(charges, pmt_ids, pmt_npes)
            
            event = {
                "charges": charges,
                "truth": events_nre[i]['truth'],
                "pmt_positions": pmt_positions,
                "hits": events_nre[i]['hits'],
                "total_charge": events_nre[i]['total_charge']
            }
            events.append(event)
            
        return events
