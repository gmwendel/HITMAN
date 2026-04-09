import numpy as np
import uproot
from hitman.tools.ratextract import DataExtractor

class PoissonDataExtractor(DataExtractor):
    def __init__(self, input_files):
        super().__init__(input_files)

    def get_poisson_train_data(self):
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
        
        hypdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcke', 'scintmod_scat_len', 'scintmod_abs_len'], library='np')
        
        charge_hyp = np.stack([hypdata['mcke'].astype(np.float32),
                               hypdata['scintmod_scat_len'].astype(np.float32),
                               hypdata['scintmod_abs_len'].astype(np.float32)
                               ], axis=1)
        
        N_events = len(charge_hyp)
        
        # Aggregate event hits into a static array (N_events, N_sensors)
        # where the value is the total charge observed by that specific PMT
        charges = np.zeros((N_events, N_sensors), dtype=np.float32)
        
        for i in range(N_events):
            pmt_ids = obsdata['mcPMTID'][i]
            pmt_npes = obsdata['mcPMTNPE'][i]
            # Accumulate charge for each PMT, defaults to 0
            np.add.at(charges[i], pmt_ids, pmt_npes)
            
        return charges, charge_hyp, pmt_positions

    def get_poisson_reco_data(self):
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
        
        hypdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcke', 'scintmod_scat_len', 'scintmod_abs_len'], library='np')
            
        charge_hyp = np.stack([hypdata['mcke'].astype(np.float32),
                               hypdata['scintmod_scat_len'].astype(np.float32),
                               hypdata['scintmod_abs_len'].astype(np.float32)
                               ], axis=1)
                               
        N_events = len(charge_hyp)
        events = []
        
        for i in range(N_events):
            charges = np.zeros(N_sensors, dtype=np.float32)
            pmt_ids = obsdata['mcPMTID'][i]
            pmt_npes = obsdata['mcPMTNPE'][i]
            np.add.at(charges, pmt_ids, pmt_npes)
            
            event = {
                "charges": charges,
                "truth": charge_hyp[i],
                "pmt_positions": pmt_positions
            }
            events.append(event)
            
        return events
