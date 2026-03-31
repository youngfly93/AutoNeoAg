from __future__ import annotations

import pandas as pd

from autoneoag.ingest.full import build_task_level_dataset


def test_build_task_level_dataset_routes_source_roles() -> None:
    source_manifest = pd.DataFrame(
        [
            {
                "source_id": "neo_train",
                "source_name": "Train Source",
                "split_role": "train_candidate",
                "source_priority": 10,
                "download_method": "public",
                "label_strength": "tier_a",
            },
            {
                "source_id": "neo_confirm",
                "source_name": "Confirm Source",
                "split_role": "confirm_candidate",
                "source_priority": 20,
                "download_method": "manual_import",
                "label_strength": "tier_a",
            },
            {
                "source_id": "neo_blind",
                "source_name": "Blind Source",
                "split_role": "blind_only",
                "source_priority": 90,
                "download_method": "manual_import",
                "label_strength": "external_blind",
            },
        ]
    )
    standardized_frames = [
        pd.DataFrame(
            [
                {
                    "peptide_mut": "YLQPRTFLL",
                    "peptide_wt": "YLQPRTFVL",
                    "hla": "HLA-A*02:01",
                    "gene": "KRAS",
                    "aa_change": "G12D",
                    "study_id": "STUDY-001",
                    "patient_id": "P001",
                    "assay_type": "ELISpot",
                    "label": 1,
                    "label_tier": "A",
                    "source_name": "Train Source",
                    "source_year": 2024,
                    "is_tesla": 0,
                    "is_simulated": 0,
                    "is_mouse": 0,
                    "mutation_event": "KRAS:G12D",
                    "peptide_length": 9,
                    "source_id": "neo_train",
                },
                {
                    "peptide_mut": "GLCTLVAML",
                    "peptide_wt": "GLCTLVAMM",
                    "hla": "HLA-A*02:01",
                    "gene": "EGFR",
                    "aa_change": "L858R",
                    "study_id": "STUDY-002",
                    "patient_id": "P002",
                    "assay_type": "FACS",
                    "label": 0,
                    "label_tier": "A",
                    "source_name": "Confirm Source",
                    "source_year": 2023,
                    "is_tesla": 0,
                    "is_simulated": 0,
                    "is_mouse": 0,
                    "mutation_event": "EGFR:L858R",
                    "peptide_length": 9,
                    "source_id": "neo_confirm",
                },
                {
                    "peptide_mut": "KLVALGINAV",
                    "peptide_wt": "KLVALGINAI",
                    "hla": "HLA-A*02:01",
                    "gene": "PIK3CA",
                    "aa_change": "H1047R",
                    "study_id": "STUDY-003",
                    "patient_id": "P003",
                    "assay_type": "ELISpot",
                    "label": 1,
                    "label_tier": "A",
                    "source_name": "Blind Source",
                    "source_year": 2025,
                    "is_tesla": 0,
                    "is_simulated": 0,
                    "is_mouse": 0,
                    "mutation_event": "PIK3CA:H1047R",
                    "peptide_length": 10,
                    "source_id": "neo_blind",
                },
            ]
        )
    ]

    dataset, source_index = build_task_level_dataset(standardized_frames, source_manifest, num_folds=4)

    split_map = dict(zip(dataset["source_id"], dataset["split"], strict=True))
    assert split_map["neo_train"] == "dev"
    assert split_map["neo_confirm"] == "confirm"
    assert split_map["neo_blind"] == "blind"
    assert dataset.loc[dataset["source_id"] == "neo_train", "fold"].iloc[0] in {0, 1, 2, 3}
    assert dataset.loc[dataset["source_id"] != "neo_train", "fold"].tolist() == [-1, -1]
    assert dataset["sample_uid"].nunique() == len(dataset)
    assert set(source_index["split"]) == {"blind", "confirm", "dev"}
