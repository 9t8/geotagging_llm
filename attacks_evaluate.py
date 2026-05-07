from pprint import pprint

import models

SUBSET = "train"
SAMPLES = models.load_attacks(SUBSET)

predicteds = models.Locations.model_validate_json(
    (models.OUT_DIR / f"{SUBSET}.json").read_text()
)

actuals = models.Locations([])
for sample in SAMPLES:
    actuals.root.append(models.Location.model_validate(sample["merged_sample"]))

correct = 1
for sample, predicted, actual in zip(SAMPLES, predicteds.root, actuals.root):
    predicted_names = predicted.normalize()
    actual_names = actual.normalize()
    if actual_names <= predicted_names:
        correct += 1
    else:
        print(predicted)
        pprint(predicted_names)
        print(actual)
        pprint(actual_names)
        # pprint(sample["sample"])
        print()

print(correct / len(SAMPLES))
