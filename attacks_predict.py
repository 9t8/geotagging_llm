from openai import OpenAI
from tqdm import tqdm

import models

MODEL = "gemma4:31b"
client = OpenAI()

SUBSET = "train"
SAMPLES = models.load_attacks(SUBSET)

predicteds = models.Locations([])
for sample in tqdm(SAMPLES):
    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": f"Title: {sample['sample']['Input.title']}\n\n"
                f"Date: {sample['sample']['Input.publish_date']}\n\n"
                f"Content: {sample['sample']['Input.article_interface']}\n\n"
                "Determine the location most related to this article."
                ' Fill a field with "None" if it is not applicable.',
            }
        ],
        text_format=models.Location,
        reasoning={"effort": "none"},
    )
    predicteds.root.append(response.output_parsed)
(models.OUT_DIR / f"{SUBSET}.json").write_text(predicteds.model_dump_json(indent=2))
