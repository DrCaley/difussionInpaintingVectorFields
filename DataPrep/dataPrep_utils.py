def normalize_data(data_set, global_min, global_max):
    margin_min = global_min - (global_min*.2)
    margin_max = global_max + (global_max*.2)
    return (data_set - margin_min) / (margin_max - margin_min)
