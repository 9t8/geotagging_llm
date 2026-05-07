import numpy as np
from haversine import haversine_vector

import models

MAX_ERROR_MI = 10

predicteds = models.ToponymsList.model_validate_json(
    (models.OUT_DIR / "lgl.json").read_text()
)

actuals = models.ToponymsList([])
for article in models.LGL_ROOT:
    toponyms = models.Toponyms([])
    for toponym in article.find("toponyms"):
        gaztag = toponym.find("gaztag")
        if gaztag is None:
            continue

        toponyms.root.append(
            models.Toponym(
                phrase=toponym.find("phrase").text,
                latitude=gaztag.find("lat").text,
                longitude=gaztag.find("lon").text,
            )
        )

    actuals.root.append(toponyms)

print("Recognition")
predictions = 0
num_actual = 0
true_positives = 0
for predicted, actual in zip(predicteds.root, actuals.root):
    pred_phrases = predicted.normalized_phrases()
    act_phrases = actual.normalized_phrases()

    predictions += len(pred_phrases)
    num_actual += len(act_phrases)
    true_positives += len(pred_phrases & act_phrases)

print(f"{predictions = }")
print(f"{num_actual = }")
print(f"{true_positives = }")
precision = true_positives / predictions
recall = true_positives / num_actual
print(f"{precision = }")
print(f"{recall = }")
print(f"F1: {2 * precision * recall / (precision + recall)}")
print()

print("Resolution")
predictions = 0
num_actual = 0
correct_predictions = 0
recalled = 0
for predicted, actual in zip(predicteds.root, actuals.root):
    pred_coords = predicted.coordinates()
    act_coords = actual.coordinates()

    predictions += len(pred_coords)
    num_actual += len(act_coords)
    if not (pred_coords and act_coords):
        continue
    proximities = (
        haversine_vector(pred_coords, act_coords, "mi", comb=True) <= MAX_ERROR_MI
    )
    correct_predictions += np.count_nonzero(np.logical_or.reduce(proximities))
    recalled += np.count_nonzero(np.logical_or.reduce(proximities, 1))

print(f"{predictions = }")
print(f"{num_actual = }")
print(f"{correct_predictions = }")
print(f"{recalled = }")
precision = correct_predictions / predictions
recall = recalled / num_actual
print(f"{precision = }")
print(f"{recall = }")
