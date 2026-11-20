import pandas as pd

from slicereg.region_groups import (FUNCTION_ORDER, MAJOR_ORDER, annotate_region_groups,
                                    count_by_group, function_class_for, major_division_for)


def test_major_division_picks_deepest_parent():
    major_by_id = {688: "Cortex", 313: "Midbrain", 1065: "Brainstem"}
    cortex = {"id": 985, "structure_id_path": [997, 8, 567, 688, 500, 985]}
    mid = {"id": 1, "structure_id_path": [997, 8, 343, 313, 1]}
    hind = {"id": 136, "structure_id_path": [997, 8, 343, 1065, 354, 136]}
    assert major_division_for(cortex, major_by_id) == "Cortex"
    assert major_division_for(mid, major_by_id) == "Midbrain"
    assert major_division_for(hind, major_by_id) == "Brainstem"
    assert major_division_for({"id": 0, "structure_id_path": [997]}, major_by_id) == "Other"


def test_function_class_prefers_sensory():
    assert function_class_for(10, {10, 11}, {11, 12}) == "Sensory"
    assert function_class_for(12, {10, 11}, {11, 12}) == "Motor"
    assert function_class_for(99, {10}, {12}) == "Other"


def test_annotate_region_groups_with_allen_atlas():
    cells = pd.DataFrame({
        "region_id": [136, 985, 429, 0],
        "region_acronym": ["IRN", "MOp", "SPVC", "outside"],
        "region_name": ["Intermediate", "Primary motor", "Spinal trigeminal", "Outside"],
        "hemisphere": ["left", "left", "right", ""],
        "cell_type": ["rfp", "rfp", "gfp", "rfp"],
    })
    out = annotate_region_groups(cells, "allen_mouse_25um")
    by = dict(zip(out["region_acronym"], zip(out["major_division"], out["function_class"])))
    assert by["IRN"] == ("Brainstem", "Other")
    assert by["MOp"][0] == "Cortex" and by["MOp"][1] == "Motor"
    assert by["SPVC"][0] == "Brainstem" and by["SPVC"][1] == "Sensory"
    assert by["outside"] == ("Other", "Other")

    major = count_by_group(out, "major_division", MAJOR_ORDER)
    assert int(major.loc[major["group"] == "Brainstem", "cells"].iloc[0]) == 2
    assert int(major.loc[major["group"] == "Cortex", "cells"].iloc[0]) == 1
    fun = count_by_group(out, "function_class", FUNCTION_ORDER)
    assert list(fun["group"]) == list(FUNCTION_ORDER)
    assert int(fun.loc[fun["group"] == "Motor", "cells"].iloc[0]) == 1
    assert int(fun.loc[fun["group"] == "Sensory", "cells"].iloc[0]) == 1
