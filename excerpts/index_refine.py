@staticmethod
def _validate_map_refines(fine_name: str, fine: list[int], coarse_name: str, coarse: list[int]) -> None:
    # Under the segmented assumption, coarse[fine[l]] == coarse[l] is equivalent to every segment of fine
    # falling inside a single segment of coarse.
    for layer_id, fine_src in enumerate(fine):
        assert coarse[fine_src] == coarse[layer_id], (
            f"shared {fine_name} map must refine {coarse_name} map: layer {layer_id} -> {fine_name} source {fine_src}, "
            f"but {coarse_name} sources differ: {coarse[layer_id]} != {coarse[fine_src]}"
        )
