def join_tracks(df, parameters, join_time_threshold=960, join_dist_threshold=20.0,
                buffer_dist=10.0, buffer_time=660, sample_cells=5):
    """
    Join similar tracks that are likely continuations of each other.
    Uses the sophisticated logic from the original best_track implementation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with track data
    parameters : pandas.DataFrame
        Track parameters from calculate_theil_sen_parameters
    join_time_threshold : float
        Maximum time gap (seconds) between tracks to consider joining (default: 16 min)
    join_dist_threshold : float
        Maximum distance (km) between track endpoints to consider joining (default: 50 km)
    buffer_dist : float
        Distance threshold for prediction accuracy check (km)
    buffer_time : float
        Time threshold for velocity difference check (seconds, default: 11 min)
    sample_cells : int
        Number of cells from later track to test prediction accuracy (default: 5)
    
    Returns:
    --------
    pandas.DataFrame
        Updated DataFrame with joined tracks
    dict
        Statistics about joining: joined_tracks
    """
    df_updated = df.copy()
    time_epoch = df['timestamp'].min()
    
    # Convert parameters to dict
    params_dict = parameters.set_index('track_id').to_dict('index')
    for track_id in params_dict:
        params_dict[track_id]['t0_seconds'] = (params_dict[track_id]['t0'] - time_epoch).total_seconds()
    
    # Get list of tracks sorted by start time
    track_ids = list(parameters['track_id'].values)
    track_starts = [(tid, params_dict[tid]['t0']) for tid in track_ids]
    track_starts.sort(key=lambda x: x[1])
    sorted_track_ids = [tid for tid, _ in track_starts]
    
    # Track which tracks have been merged
    merged_into = {}  # Maps track_id -> track_id it was merged into
    tracks_removed = set()
    join_count = 0
    
    for i, track1_id in enumerate(sorted_track_ids):
        # Skip if this track was already merged
        if track1_id in tracks_removed:
            continue
        
        track1_data = df_updated[df_updated['track_id'] == track1_id]
        
        # Skip tracks with only 1 cell
        if len(track1_data) < 2:
            continue
        
        track1_params = params_dict[track1_id]
        
        # Get track1 end time and position
        track1_end = track1_data['timestamp'].max()
        track1_xf = track1_data.iloc[-1]['x']
        track1_yf = track1_data.iloc[-1]['y']
        
        # Compare with earlier tracks
        for j in range(i):
            track2_id = sorted_track_ids[j]
            
            # Skip if track2 was already merged
            if track2_id in tracks_removed:
                continue
            
            track2_data = df_updated[df_updated['track_id'] == track2_id]
            
            # Skip tracks with only 1 cell
            if len(track2_data) < 2:
                continue
            
            track2_params = params_dict[track2_id]
            
            # Determine which track is earlier and which is later
            if track1_params['t0'] > track2_params['t0']:
                early_id = track2_id
                late_id = track1_id
                early_params = track2_params
                late_params = track1_params
                early_data = track2_data
                late_data = track1_data
            else:
                early_id = track1_id
                late_id = track2_id
                early_params = track1_params
                late_params = track2_params
                early_data = track1_data
                late_data = track2_data
            
            # Get early track end time and position
            early_tend = early_data['timestamp'].max()
            early_xf = early_data.iloc[-1]['x']
            early_yf = early_data.iloc[-1]['y']
            
            # Get late track start position
            late_t0 = late_params['t0']
            late_x0 = late_data.iloc[0]['x']
            late_y0 = late_data.iloc[0]['y']
            
            # Check 1: Time gap between tracks
            time_gap = (late_t0 - early_tend).total_seconds()
            if abs(time_gap) > join_time_threshold:
                continue
            
            # Check 2: Distance between track endpoints
            endpoint_dist = np.sqrt((early_xf - late_x0)**2 + (early_yf - late_y0)**2)
            if endpoint_dist > join_dist_threshold:
                continue
            
            # Check 3: Velocity difference between tracks
            # Velocity should be similar for tracks to be joined
            velocity_diff = np.sqrt((early_params['u'] - late_params['u'])**2 + 
                                   (early_params['v'] - late_params['v'])**2)
            max_velocity_diff = buffer_dist / buffer_time  # km/s
            if velocity_diff > max_velocity_diff:
                continue
            
            # Check 4: Prediction accuracy
            # Test if early track's trajectory predicts late track's first cells
            late_sorted = late_data.sort_values('timestamp')
            test_cells = late_sorted.head(min(sample_cells, len(late_sorted)))
            
            prediction_errors = []
            for idx, cell in test_cells.iterrows():
                cell_time_seconds = (cell['timestamp'] - time_epoch).total_seconds()
                early_tend_seconds = (early_tend - time_epoch).total_seconds()
                dt = cell_time_seconds - early_tend_seconds
                
                # Predict position using early track's trajectory from its endpoint
                x_predict = early_xf + early_params['u'] * dt
                y_predict = early_yf + early_params['v'] * dt
                
                # Calculate prediction error
                error = np.sqrt((cell['x'] - x_predict)**2 + (cell['y'] - y_predict)**2)
                prediction_errors.append(error)
            
            mean_prediction_error = np.mean(prediction_errors)
            if mean_prediction_error > buffer_dist:
                continue
            
            # All checks passed - join the tracks!
            # Merge late track into early track
            tracks_removed.add(late_id)
            merged_into[late_id] = early_id
            join_count += 1
        
    # Apply merges
    for old_id, new_id in merged_into.items():
        df_updated.loc[df_updated['track_id'] == old_id, 'track_id'] = new_id
        df_updated.loc[df_updated['track_id'] == old_id, 'filter'] = 2 # Mark as merged in filter column
    
    return df_updated, join_count
