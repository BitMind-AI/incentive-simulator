

def average_challenges_per_tempo(df, timestamp_col='_timestamp', num_blocks=360, block_s=12):

    start_time = df[timestamp_col].min()
    end_time = df[timestamp_col].max()

    tempo_s = num_blocks * block_s
    num_tempos = (end_time - start_time) // tempo_s + 1
    counts_per_tempo = []
    
    for i in range(int(num_tempos)):
        tempo_start = start_time + i * tempo_s
        tempo_end = tempo_start + tempo_s
        count = df[(df[timestamp_col] >= tempo_start) & (df[timestamp_col] < tempo_end)].shape[0]
        counts_per_tempo.append(count)
    
    avg_chalalenges = sum(counts_per_tempo) / num_tempos
    return avg_chalalenges, counts_per_tempo