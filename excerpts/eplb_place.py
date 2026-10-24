rank = scores.argmin(dim=-1)
occupied = occupancy[layers, rank]
indices[layers, rank * capacity + occupied] = physical_order[:, job_index]
occupancy[layers, rank] = occupied + 1
updated = loads[layers, rank] + vector
loads[layers, rank] = updated
torch.maximum(peaks, updated, out=peaks)
