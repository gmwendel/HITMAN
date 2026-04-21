import numpy as np
import uproot
from hitman.tools.ratextract import DataExtractor

class FactorizedDataExtractor(DataExtractor):
    def __init__(self, input_files=None):
        if input_files is not None:
            super().__init__(input_files)

    def _process_charges(self, charges, injected_yields):
        """
        Takes raw charge (hits) arrays of shape (N_events, N_sensors) and converts them into:
        1. shape_target: the normalized PMF across all sensors for each event (no artificial smoothing).
        2. rate_target: the absolute integer hits (K_sim) and the injected photons (eta_sim).
        """
        rate_target = np.stack([np.sum(charges, axis=1), injected_yields], axis=1)
        
        # Normalize to create the exact probability mass function (PMF) where sum(sensors) = 1.0
        # No Laplace smoothing applied so the network targets the true 0's
        event_totals = np.sum(charges, axis=1, keepdims=True)
        # Prevent division by zero for events with 0 hits (though they are filtered out elsewhere)
        shape_target = charges / np.clip(event_totals, 1e-12, None)
        
        return shape_target, rate_target

    def get_factorized_train_data(self):
        # First, grab the standard hit data
        _, hit_obs, charge_hyp_old, hit_hyp = super().get_hitman_train_data()
        
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcPMTNPE', 'mcPMTID'], library='np')
        maps = uproot.concatenate([self.input_files[0] + ":meta;1"],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')
        hypdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['scintPhotons', 'cherPhotons', 'scintmod_scat_len', 'scintmod_abs_len', 'mcx', 'mcy', 'mcz'], library='np')
        
        pmt_positions = np.stack([
            maps['pmtX'][0].astype(np.float32),
            maps['pmtY'][0].astype(np.float32),
            maps['pmtZ'][0].astype(np.float32)
        ], axis=1)
        
        injected_yields = hypdata['scintPhotons'].astype(np.float32) # TODO: Include cherPhotons in future work once they are properly scaled with respect to light yield and detector sensitivity.
        charge_hyp = np.stack([hypdata['scintmod_scat_len'].astype(np.float32),
                               hypdata['scintmod_abs_len'].astype(np.float32)
                               ], axis=1)
        
        vertex = np.stack([hypdata['mcx'].astype(np.float32),
                           hypdata['mcy'].astype(np.float32),
                           hypdata['mcz'].astype(np.float32)], axis=1)
        
        N_sensors = len(pmt_positions)
        N_events = len(charge_hyp)
        
        event_lengths = np.array([len(x) for x in obsdata['mcPMTID']], dtype=np.int32)
        event_indices = np.repeat(np.arange(N_events), event_lengths)
        flat_pmt_ids = np.concatenate(obsdata['mcPMTID']).astype(np.int32)
        flat_npes = np.concatenate(obsdata['mcPMTNPE'])
        
        charges = np.zeros((N_events, N_sensors), dtype=np.float32)
        np.add.at(charges, (event_indices, flat_pmt_ids), flat_npes)
        
        shape_target, rate_target = self._process_charges(charges, injected_yields)
            
        return shape_target, rate_target, charge_hyp, pmt_positions, vertex, hit_obs, hit_hyp

    def get_factorized_only_train_data(self):
        # Optimized loading: exclusively load necessary arrays, entirely skipping the massive mcPEFrontEndTime
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcPMTNPE', 'mcPMTID'], library='np')
        maps = uproot.concatenate([self.input_files[0] + ":meta;1"],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')
        hypdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['scintPhotons', 'cherPhotons', 'scintmod_scat_len', 'scintmod_abs_len', 'mcx', 'mcy', 'mcz'], library='np')
        
        pmt_positions = np.stack([
            maps['pmtX'][0].astype(np.float32),
            maps['pmtY'][0].astype(np.float32),
            maps['pmtZ'][0].astype(np.float32)
        ], axis=1)
        
        injected_yields = hypdata['scintPhotons'].astype(np.float32) # TODO: Include cherPhotons in future work once they are properly scaled with respect to light yield and detector sensitivity.
        charge_hyp = np.stack([hypdata['scintmod_scat_len'].astype(np.float32),
                               hypdata['scintmod_abs_len'].astype(np.float32)
                               ], axis=1)
        
        vertex = np.stack([hypdata['mcx'].astype(np.float32),
                           hypdata['mcy'].astype(np.float32),
                           hypdata['mcz'].astype(np.float32)], axis=1)

        N_sensors = len(pmt_positions)
        N_events = len(charge_hyp)
        
        # Vectorized charge accumulation (bypassing slow Python loops over N_events)
        event_lengths = np.array([len(x) for x in obsdata['mcPMTID']], dtype=np.int32)
        event_indices = np.repeat(np.arange(N_events), event_lengths)
        flat_pmt_ids = np.concatenate(obsdata['mcPMTID']).astype(np.int32)
        flat_npes = np.concatenate(obsdata['mcPMTNPE'])
        
        charges = np.zeros((N_events, N_sensors), dtype=np.float32)
        np.add.at(charges, (event_indices, flat_pmt_ids), flat_npes)
        
        shape_target, rate_target = self._process_charges(charges, injected_yields)
            
        return shape_target, rate_target, charge_hyp, pmt_positions, vertex

    def get_factorized_reco_data(self):
        events_nre = super().get_hitman_reco_data()
        
        obsdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['mcPMTNPE', 'mcPMTID'], library='np')
        maps = uproot.concatenate([self.input_files[0] + ":meta;1"],
                                  filter_name=["pmtX", "pmtY", "pmtZ"], library='np')
        hypdata = uproot.concatenate(
            [self.input_files[i] + ":" + self.out_keys[i] for i in range(len(self.input_files))],
            filter_name=['scintPhotons', 'cherPhotons', 'mcx', 'mcy', 'mcz'], library='np')
        
        pmt_positions = np.stack([
            maps['pmtX'][0].astype(np.float32),
            maps['pmtY'][0].astype(np.float32),
            maps['pmtZ'][0].astype(np.float32)
        ], axis=1)
        
        injected_yields = hypdata['scintPhotons'].astype(np.float32) # TODO: Include cherPhotons in future work once they are properly scaled with respect to light yield and detector sensitivity.
        
        vertex = np.stack([hypdata['mcx'].astype(np.float32),
                           hypdata['mcy'].astype(np.float32),
                           hypdata['mcz'].astype(np.float32)], axis=1)
        
        N_sensors = len(pmt_positions)
        N_events = len(events_nre)
        events = []
        
        for i in range(N_events):
            charges = np.zeros(N_sensors, dtype=np.float32)
            pmt_ids = obsdata['mcPMTID'][i]
            pmt_npes = obsdata['mcPMTNPE'][i]
            np.add.at(charges, pmt_ids, pmt_npes)
            
            # Note: For reconstruction, we pass the raw charges out. 
            # The likelihood function will do the math against the network outputs.
            event = {
                "charges": charges,
                "truth": events_nre[i]['truth'],
                "vertex": vertex[i],
                "pmt_positions": pmt_positions,
                "hits": events_nre[i]['hits'],
                "total_charge": events_nre[i]['total_charge'],
                "injected_yields": injected_yields[i]
            }
            events.append(event)
            
        return events