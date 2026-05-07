import json
import re
from pathlib import Path

import pydantic
from defusedxml.ElementTree import parse

OUT_DIR = Path("build")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(name: str) -> str:
    return re.sub(
        r"[\(\[\{].*?[\)\]\}]| of america|people's republic of | and herzegovina"
        r"| state$| province$|[^0-9a-z]",
        r"",
        re.sub(
            r"UK|England", r"United Kingdom", re.sub(r"USA?", r"United States", name)
        ).lower(),
    )


LGL_ROOT = parse(
    "Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/lgl.xml"
).getroot()


class Toponym(pydantic.BaseModel):
    phrase: str
    latitude: float
    longitude: float


class Toponyms(pydantic.RootModel[list[Toponym]]):
    def normalized_phrases(self) -> set:
        return {normalize(toponym.phrase) for toponym in self.root}

    def coordinates(self) -> list:
        return [(toponym.latitude, toponym.longitude) for toponym in self.root]


class ToponymsList(pydantic.RootModel[list[Toponyms]]):
    pass


def load_attacks(subset: str) -> dict:
    return json.loads(Path(f"HRDsAttack/data/{subset}.json").read_text())


class Location(pydantic.BaseModel):
    country: str
    region: str
    city: str

    def normalize(self) -> set:
        return {
            normalize(self.country),
            normalize(self.region),
            normalize(self.city),
        } - {"none", "unknown"}


class Locations(pydantic.RootModel[list[Location]]):
    pass
