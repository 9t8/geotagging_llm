from openai import OpenAI
from tqdm import tqdm

import models

MODEL = "gemma4:31b"
client = OpenAI()

predicteds = models.ToponymsList([])
for article in tqdm(models.LGL_ROOT):
    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": f"Title: {article.find('title').text}\n\n"
                f"Domain: {article.find('domain').text}\n\n"
                f"Content: {article.find('text').text}\n\n"
                "Detect and resolve the toponyms in the content. Include only"
                " countries, regions, states, cities, land or water features, and"
                " parks.",
            }
        ],
        text_format=models.Toponyms,
        reasoning={"effort": "none"},
    )
    predicteds.root.append(response.output_parsed)

(models.OUT_DIR / "lgl.json").write_text(predicteds.model_dump_json(indent=2))
