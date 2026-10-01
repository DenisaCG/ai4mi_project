"""Validate patient-level holdout and equal-sized cross-validation splits."""


def validate_split(patient_count: int, retains: int, fold: int,
                   num_folds: int | None = None) -> None:
    if type(retains) is not int or not 0 < retains < patient_count:
        raise ValueError(f"retains must be between 1 and {patient_count - 1}")
    if type(fold) is not int or fold < 0:
        raise ValueError("fold must be a non-negative integer index")
    if num_folds is not None:
        if type(num_folds) is not int or not 2 <= num_folds <= patient_count:
            raise ValueError(f"num_folds must be between 2 and {patient_count}")
        if retains * num_folds != patient_count:
            raise ValueError(
                f"retains ({retains}) × num_folds ({num_folds}) "
                f"must equal the patient count ({patient_count})")
        if fold >= num_folds:
            raise ValueError(f"fold index must be between 0 and {num_folds - 1}")
    if (fold + 1) * retains > patient_count:
        raise ValueError("fold index selects fewer than retains validation patients")
