from slicereg.render import resolve_regions, split_region_text, suggest_regions

STRUCTURES = [
    {"acronym": "IRN", "name": "Intermediate reticular nucleus"},
    {"acronym": "PARN", "name": "Parvicellular reticular nucleus"},
    {"acronym": "XII", "name": "Hypoglossal nucleus"},
    {"acronym": "V", "name": "Motor nucleus of trigeminal"},
    {"acronym": "P5", "name": "Peritrigeminal zone"},
    {"acronym": "PERI5", "name": "Perirhinal area, layer 5"},
    {"acronym": "Acs5", "name": "Accessory trigeminal nucleus"},
    {"acronym": "VII", "name": "Facial motor nucleus"},
]


def test_split_region_text():
    assert split_region_text("irt/pcrt xii, mo5;peri5  ") == ["irt", "pcrt", "xii", "mo5",
                                                              "peri5"]


def test_paxinos_names_resolve_to_allen():
    found, missing = resolve_regions(split_region_text("irt/pcrt xii mo5 peri5 acc5 fmn"),
                                     STRUCTURES)
    assert [acr for _, acr, _ in found] == ["IRN", "PARN", "XII", "V", "P5", "Acs5", "VII"]
    assert missing == []


def test_allen_acronyms_case_insensitive_and_deduplicated():
    found, missing = resolve_regions(["irn", "IRt", "vii", "nope"], STRUCTURES)
    assert [acr for _, acr, _ in found] == ["IRN", "VII"]
    assert missing == ["nope"]


def test_suggestions():
    assert "IRN" in suggest_regions("reticular", STRUCTURES)
