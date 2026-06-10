LABEL_NAMES = ["clean", "toxic"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}
NUM_LABELS = len(LABEL_NAMES)
