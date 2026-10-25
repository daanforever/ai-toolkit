"""Download 10 Commons photos into ./datasets without cropping or resizing."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "ai-toolkit-dataset/1.0 (local training dataset)"}
OUT = Path("datasets")
CATEGORIES = [
    "Category:Featured_pictures_of_landscapes",
    "Category:Featured_pictures_of_architecture",
    "Category:Featured_pictures_of_cities",
    "Category:Featured_pictures_of_flowers",
    "Category:Featured_pictures_of_mountains",
    "Category:Featured_pictures_of_bridges",
    "Category:Featured_pictures_of_waterfalls",
    "Category:Featured_pictures_of_forests",
    "Category:Featured_pictures_of_beaches",
    "Category:Featured_pictures_of_birds",
]


def get_json(params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        "https://commons.wikimedia.org/w/api.php?" + query,
        headers=UA,
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp)


def candidates(category: str) -> list[dict]:
    data = get_json(
        {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": "40",
            "prop": "imageinfo",
            "iiprop": "size|mime|url",
            "format": "json",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    found = []
    for page in pages.values():
        info = page["imageinfo"][0]
        width = int(info["width"])
        height = int(info["height"])
        short = min(width, height)
        long = max(width, height)
        if info.get("mime") != "image/jpeg":
            continue
        if short < 1024 or long < 1024:
            continue
        if not (0.5 <= width / height <= 2.0):
            continue
        if long == width:
            thumb_w = 2048 if width >= 2048 else width
        elif width >= height:
            thumb_w = 2048 if width >= 2048 else width
        else:
            thumb_w = int(round(2048 * width / height)) if height >= 2048 else width
        found.append(
            {
                "title": page["title"],
                "width": width,
                "height": height,
                "thumb_w": max(1, thumb_w),
            }
        )
    return found


def thumb_url(title: str, thumb_w: int) -> tuple[str, int, int]:
    data = get_json(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": str(thumb_w),
            "format": "json",
        }
    )
    page = next(iter(data["query"]["pages"].values()))
    info = page["imageinfo"][0]
    return info["thumburl"], int(info["thumbwidth"]), int(info["thumbheight"])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    picked: list[dict] = []
    seen: set[str] = set()
    for category in CATEGORIES:
        if len(picked) >= 10:
            break
        for item in candidates(category):
            if item["title"] in seen:
                continue
            seen.add(item["title"])
            picked.append(item)
            break
    if len(picked) < 10:
        raise SystemExit(f"only found {len(picked)} images")

    for index, item in enumerate(picked, 1):
        url, tw, th = thumb_url(item["title"], item["thumb_w"])
        short = min(tw, th)
        long = max(tw, th)
        if short < 1024 or long > 2048 or not (0.5 <= tw / th <= 2.0):
            raise SystemExit(
                f"thumb outside range {tw}x{th} for {item['title']}"
            )
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
        path = OUT / f"{index:03d}.jpg"
        path.write_bytes(payload)
        print(f"{path.name} {tw}x{th} {item['title']}")


if __name__ == "__main__":
    main()
