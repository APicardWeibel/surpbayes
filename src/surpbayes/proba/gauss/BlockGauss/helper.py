def check_blocks(blocks: list[list[int]]):
    items = [i for group in blocks for i in group]
    set_items = set(items)
    n_tot = len(set_items)

    if n_tot != len(items):
        raise ValueError("An index should be in only one group")

    if set_items != set(range(n_tot)):
        raise ValueError(f"All indexes between 0 and {n_tot} should belong to a group")
