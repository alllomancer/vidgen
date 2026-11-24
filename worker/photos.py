"""Scene list and prompts for `photos` jobs: one person placed in several scenes.

Pure Python (no torch) so it can be checked anywhere:
    python3 worker/photos.py
"""

import random

MAX_PHOTOS = 8

# Output sizes (width, height), about 1 MP, multiples of 16.
ASPECTS = {"portrait": (896, 1152), "landscape": (1152, 896), "square": (1024, 1024)}

# Random fill: people-friendly settings with distinct light and backgrounds.
LOCATIONS = [
    "a sunlit sidewalk café in Paris",
    "a busy Tokyo street at night with neon signs",
    "a white sand beach at golden hour",
    "a mountain trail in the Swiss Alps",
    "a cozy bookstore with wooden shelves",
    "a modern open-plan office with large windows",
    "a farmers market with colorful fruit stalls",
    "a misty pine forest in the morning",
    "a rooftop terrace overlooking a city skyline at dusk",
    "a narrow cobblestone street in Rome",
    "a Moroccan spice market",
    "a quiet library reading room",
    "a lakeside wooden dock at sunrise",
    "a New York subway platform",
    "a vineyard in Tuscany in late afternoon",
    "a snowy street in a small alpine village",
    "a desert canyon in Utah",
    "a botanical garden greenhouse",
    "a London park in autumn with fallen leaves",
    "a minimalist Scandinavian living room",
    "a seaside harbor with fishing boats",
    "a busy airport terminal",
    "a coffee roastery with exposed brick walls",
    "a tropical waterfall in a rainforest",
    "a city bridge over a river at blue hour",
    "a lavender field in Provence",
    "a retro diner with red booths",
    "a modern art museum gallery",
    "a college campus lawn on a sunny day",
    "a bustling train station concourse",
    "a hillside overlooking Santorini's white houses",
    "a ski lodge with a stone fireplace",
    "a street food market in Bangkok at night",
    "a cherry blossom path in Kyoto",
    "a sunny kitchen with marble countertops",
    "a yoga studio with wooden floors and plants",
    "a rocky Icelandic coastline with black sand",
    "a jazz bar with warm dim lighting",
    "a sailboat deck on open blue water",
    "a quiet suburban street with autumn trees",
    "a modern hotel lobby",
    "a wheat field at sunset",
    "a graffiti-covered alley in Berlin",
    "a mountain lake with clear turquoise water",
    "a small Greek island taverna by the sea",
    "a rainy city street with reflections on the pavement",
    "a cozy cabin porch in the woods",
    "a flower shop full of fresh bouquets",
    "a tennis court on a bright morning",
    "a historic library with a vaulted ceiling",
]

IDENTITY = ("Keep this exact person identical: same face, facial features, hair, skin tone, "
            "body shape and outfit.")
# When the scene says what to wear (--wearing), the outfit is not kept from the reference.
IDENTITY_NEW_OUTFIT = ("Keep this exact person identical: same face, facial features, hair, skin tone "
                       "and body shape. Dress them as described, replacing the outfit in the reference.")
# Structured scene fields (--where/--wearing/--doing/--camera); any non-empty subset.
SHOT_FIELDS = ("where", "wearing", "doing", "camera")
PREPOSITIONS = {"in", "on", "at", "by", "under", "inside", "near", "beside", "outside", "above", "below",
                "on top", "atop", "across", "along", "behind", "in front"}
PHOTO_STYLE = ("Photorealistic photograph, shot on a full-frame camera with a 50mm lens, natural light "
               "matching the scene, realistic skin texture, accurate shadows and perspective, sharp focus.")
# Look presets (`look`); "pro" is the original wording, so jobs without options are unchanged.
LOOKS = {
    "pro": PHOTO_STYLE,
    "phone": ("Photorealistic casual smartphone photo taken by a friend, phone main camera, natural available "
              "light, slightly imperfect everyday framing, true-to-life colors, realistic skin texture, "
              "accurate shadows and perspective, sharp focus."),
    "candid": ("Photorealistic candid documentary photograph on a 35mm lens, available light only, an unposed "
               "real moment, true-to-life colors, natural depth of field, realistic skin texture, accurate "
               "shadows and perspective."),
    "portrait": ("Photorealistic professional portrait photograph, 85mm lens at f/1.8, shallow depth of field "
                 "with soft background blur, soft natural light, true-to-life skin tones, realistic skin "
                 "texture, sharp focus on the eyes."),
    "film": ("Photorealistic 35mm film photograph, Kodak Portra 400 look, soft natural light, gentle film "
             "grain, slightly muted warm colors, organic tones, realistic skin texture, accurate shadows."),
}
NATURAL_SKIN = ("Natural skin with visible pores and fine texture, small natural imperfections, no retouching, "
                "no beauty filter, no heavy makeup look.")
CANDID = "Candid, unposed moment; they are not looking at the camera."

# Series: photo 1 shows the place clearly (it anchors photos 2..N); the rest vary pose and camera.
# Each later shot says whether it gets photo 1 as a reference. Shots that look at the place from
# far away or another side do not: with photo 1 in hand the model copies its background and
# viewpoint (live check 2026-11-22), so those rely on the text and the avatar images alone.
SERIES_FIRST = ("standing naturally, full body, the place clearly visible around them",
                "eye level, medium-wide shot")
# (action, camera, anchored)
SERIES_SHOTS = [
    ("walking slowly, glancing to the side", "from the side, full body, a different part of the place behind them", False),
    ("sitting down, relaxed", "medium shot from a low angle across the table", True),
    ("laughing, looking away from the camera", "three-quarter view, waist up, background softly blurred", True),
    ("looking out at the surroundings", "from behind over the shoulder, looking the other way across the place", False),
    ("leaning against something nearby, arms crossed", "medium shot, slightly low angle, from the opposite side", True),
    ("a quiet thoughtful expression", "close-up portrait, shallow depth of field", True),
    ("mid-step, turning towards the camera", "wide shot from far across the place, the person small in the frame", False),
    ("adjusting their sleeve, looking down", "medium close-up from slightly above", True),
    ("smiling at the camera", "head and shoulders, eye level, background blurred", True),
    ("seen from a distance", "wide establishing shot from the far end of the place, high angle", False),
]
# Cameras for the user's own actions (photos 2..N), cycled.
SERIES_CAMERAS = [(camera, anchored) for _, camera, anchored in SERIES_SHOTS]
ANCHOR = ("Image {n} is an earlier photo from this same shoot. Use it only for the place's look, the "
          "lighting, the time of day and the outfit. Do not copy its composition, camera position, "
          "background framing, pose or props: the camera is somewhere else in the same place, as "
          "described below, so the background shows the place from that new angle.")
SERIES_PLACE = ("All photos in this set are from one shoot in the same place with the same outfit and light; "
                "this one shows it from a new angle.")

NEGATIVE = ("cartoon, anime, illustration, CGI, 3d render, painting, plastic skin, airbrushed skin, "
            "oversaturated, distorted face, deformed hands, extra fingers, extra limbs, blurry, "
            "watermark, text")


def location_scene(location):
    return f"in {location}, natural relaxed pose suited to the setting"


def shot_scene(shot):
    """{"where", "wearing", "doing", "camera"} (any subset) -> (scene text, keep_outfit)."""
    parts = [shot.get("doing") or "in a natural relaxed pose"]
    if shot.get("where"):
        where = shot["where"]
        lead = where.split(" ", 1)[0].lower()
        parts.append(where if lead in PREPOSITIONS else f"in {where}")
    if shot.get("wearing"):
        parts.append(f"wearing {shot['wearing']}")
    scene = ", ".join(parts)
    if shot.get("camera"):
        scene += f". Camera: {shot['camera']}"
    return scene, not shot.get("wearing")


def build_scenes(scenes, count, seed=None):
    """Final ordered list of {"source", "text", "scene"} for a job.

    scenes: [{"prompt": str} | {"location": str} | {"shot": {where, wearing, doing, camera}}].
    count: total photos; anything beyond `scenes` is a random location from
    LOCATIONS, without repeats and skipping locations the user gave.
    """
    out = []
    for s in scenes:
        if "prompt" in s:
            out.append({"source": "prompt", "text": s["prompt"], "scene": s["prompt"]})
        elif "shot" in s:
            scene, keep_outfit = shot_scene(s["shot"])
            text = "; ".join(f"{k}: {s['shot'][k]}" for k in SHOT_FIELDS if s["shot"].get(k))
            out.append({"source": "shot", "text": text, "scene": scene, "keep_outfit": keep_outfit})
        else:
            out.append({"source": "location", "text": s["location"], "scene": location_scene(s["location"])})
    given = {s["location"].strip().lower() for s in scenes if "location" in s}
    pool = [loc for loc in LOCATIONS if loc.lower() not in given]
    for loc in random.Random(seed).sample(pool, max(0, count - len(out))):
        out.append({"source": "random", "text": loc, "scene": location_scene(loc)})
    return out


def _where(where):
    lead = where.split(" ", 1)[0].lower()
    return where if lead in PREPOSITIONS else f"in {where}"


def build_series(series, count, seed=None):
    """Scenes for a series: same place and outfit, pose and camera varied.
    series: {"where", "wearing"?, "doing"?: [str]}. Photo 1 keeps the place-visible framing (the anchor);
    given actions are used in order with varied cameras, the rest are seed-sampled SERIES_SHOTS
    without repeats. Each scene says whether it is anchored on photo 1."""
    doing = list(series.get("doing") or [])
    rng = random.Random(seed)
    auto = rng.sample(SERIES_SHOTS, max(0, count - max(len(doing), 1)))
    cameras = SERIES_CAMERAS[rng.randrange(len(SERIES_CAMERAS)):] + SERIES_CAMERAS
    out = []
    for i in range(count):
        if i == 0:
            action = f"{doing[0]}, full body, the place clearly visible around them" if doing else SERIES_FIRST[0]
            camera, anchored = SERIES_FIRST[1], False
        elif i < len(doing):
            (camera, anchored), action = cameras[(i - 1) % len(cameras)], doing[i]
        else:
            action, camera, anchored = auto[i - max(len(doing), 1)]
        parts = [action, _where(series["where"])]
        if series.get("wearing"):
            parts.append(f"wearing {series['wearing']}")
        scene = ", ".join(parts) + f". Camera: {camera}"
        if i:
            scene += f". {SERIES_PLACE}"
        out.append({"source": "series", "text": f"{action} ({camera})", "scene": scene,
                    "keep_outfit": not series.get("wearing"), "anchor": anchored})
    return out


def identity(n_refs=1, keep_outfit=True):
    """Keep-identity sentence; with several reference images, says they are one person."""
    keep = IDENTITY if keep_outfit else IDENTITY_NEW_OUTFIT
    if n_refs <= 1:
        return keep
    return (f"Images 1-{n_refs} all show the same single person from different views. "
            f"{keep} Show only this one person, once.")


def photo_prompt(scene, n_refs=1, keep_outfit=True, look="pro", natural_skin=False, candid=False, anchor=False):
    """n_refs: images of the person. anchor: one more image follows them, an earlier photo of the series."""
    parts = [identity(n_refs, keep_outfit)]
    if anchor:
        parts.append(ANCHOR.format(n=n_refs + 1))
    parts.append(f"Show them {scene}.")
    if candid:
        parts.append(CANDID)
    parts.append(LOOKS[look])
    if natural_skin:
        parts.append(NATURAL_SKIN)
    return " ".join(parts)


def text_scene(scene):
    """A photo without a person (Krea 2 text-to-image): the scene as plain text."""
    if "prompt" in scene:
        return scene["prompt"]
    if "location" in scene:
        return scene["location"]
    shot = scene["shot"]
    text = ", ".join(v for v in (shot.get("doing"), shot.get("where") and _where(shot["where"])) if v)
    return text + (f". Camera: {shot['camera']}" if shot.get("camera") else "")


def scene_prompt(scene_text, look="pro", natural_skin=False, candid=False):
    """Text-to-image prompt (no reference person): the scene, then the look sentences."""
    parts = [scene_text.rstrip(". ") + "."]
    if candid:
        parts.append(CANDID)
    parts.append(LOOKS[look])
    if natural_skin:
        parts.append(NATURAL_SKIN)
    return " ".join(parts)


def add_grain(image, seed=0, strength=3.5):
    """Real film grain on the finished photo: seeded luminance noise plus a slight softening,
    so the result is not 'too clean'. Same seed and image -> same output."""
    import numpy as np
    from PIL import Image, ImageFilter

    soft = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.4))
    arr = np.asarray(soft).astype(np.float32)
    noise = np.random.default_rng(seed).normal(0.0, strength, arr.shape[:2]).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise[..., None], 0, 255).astype(np.uint8))


# Krea 2 avatar LoRAs: one Qwen photo per prompt (docs/specs/2026-11-24-krea-avatar-lora.md). Varied framing,
# light, places and outfits, so the LoRA learns the face rather than one outfit (from the 2026-11-24 spike);
# neutral wording (them/their) so it fits any avatar.
KREA_DATASET_PROMPTS = [
    'close-up portrait, looking at the camera, soft window light, plain grey wall behind them',
    'head and shoulders, three-quarter view, wearing a navy wool sweater, warm indoor lamp light',
    'in profile, looking to the left, wearing a dark green field jacket, overcast daylight on a pier',
    'medium shot, smiling slightly, wearing a white linen shirt, golden hour light in a garden',
    'full body, standing on a city sidewalk, wearing jeans and a grey hoodie, midday sun',
    'sitting at a wooden kitchen table with a mug of coffee, wearing a flannel shirt, morning light',
    'full body, walking along a beach in a light trench coat, cloudy sky, wind in their hair',
    'close-up, laughing, wearing a black t-shirt, night city lights bokeh behind them',
    'medium shot, reading a newspaper on a park bench, wearing a brown corduroy jacket, autumn leaves',
    'head and shoulders, serious expression, wearing a dark suit, office with large windows',
    'three-quarter view, hiking on a mountain trail, wearing a red outdoor jacket and backpack, bright sky',
    'close-up portrait in a grey rain jacket with the hood down, rain drops, harbour behind them',
    'full body, leaning against a brick wall, wearing a leather jacket and dark trousers, evening light',
    'medium shot, cooking at a stove, wearing a striped apron over a blue shirt, kitchen light',
    'looking over their shoulder, seen from behind and to the side, wearing a beige coat, train station',
    'sitting in an armchair by a fireplace, wearing a cardigan, warm firelight, medium shot',
    'close-up, looking slightly down, wearing a knitted beanie and scarf, snowy street, cold light',
    'medium shot, sitting in a small boat on a misty lake at dawn, wearing an olive waterproof jacket',
    'full body, riding a bicycle on a country road, wearing a polo shirt and chinos, summer afternoon',
    'head and shoulders, wearing glasses and a grey shirt, library shelves behind them, soft light',
    'three-quarter view, standing at a market stall holding apples, wearing a denim jacket, daylight',
    'medium shot, playing an acoustic guitar on a porch, wearing a plaid shirt, sunset light',
    'close-up portrait, neutral expression, studio lighting, black background, wearing a white t-shirt',
    'full body, standing on a rocky coast in a dark blue waterproof jacket, dramatic clouds',
]


if __name__ == "__main__":
    a = build_scenes([{"prompt": "reading on a train"}, {"location": "a white sand beach at golden hour"}], 6, seed=7)
    b = build_scenes([{"prompt": "reading on a train"}, {"location": "a white sand beach at golden hour"}], 6, seed=7)
    assert a == b, "same seed must give the same scenes"
    assert [s["source"] for s in a] == ["prompt", "location", "random", "random", "random", "random"]
    texts = [s["text"].lower() for s in a[1:]]
    assert len(set(texts)) == len(texts), "random fill must not repeat or reuse a given location"
    assert len(build_scenes([], MAX_PHOTOS, seed=1)) == MAX_PHOTOS
    assert len(set(LOCATIONS)) == len(LOCATIONS) >= MAX_PHOTOS
    shot = build_scenes([{"shot": {"where": "a rainy rooftop", "wearing": "a navy suit",
                                   "doing": "walking with an umbrella", "camera": "low angle, wide lens"}}], 1)[0]
    assert shot["scene"] == ("walking with an umbrella, in a rainy rooftop, wearing a navy suit. "
                             "Camera: low angle, wide lens"), shot["scene"]
    assert shot["keep_outfit"] is False and shot["source"] == "shot"
    p = photo_prompt(shot["scene"], 3, shot["keep_outfit"])
    assert "outfit." not in p.split("Show them")[0] and "replacing the outfit" in p and p.startswith("Images 1-3")
    on = build_scenes([{"shot": {"where": "on a rooftop"}}], 1)[0]["scene"]
    assert on == "in a natural relaxed pose, on a rooftop", on
    only_cam = build_scenes([{"shot": {"camera": "from directly above"}}], 1)[0]
    assert only_cam["keep_outfit"] is True and only_cam["scene"].endswith("Camera: from directly above")
    assert "body shape and outfit." in photo_prompt(only_cam["scene"])
    assert photo_prompt("x") == f"{IDENTITY} Show them x. {PHOTO_STYLE}", "no options = original prompt"
    ser = build_series({"where": "a Lisbon rooftop", "doing": ["reading", "pouring coffee"]}, 4, seed=3)
    assert ser[0]["scene"].startswith("reading, full body") and "in a Lisbon rooftop" in ser[0]["scene"]
    assert ser[1]["scene"].startswith("pouring coffee") and len({s["scene"] for s in ser}) == 4
    assert ser[0]["anchor"] is False and all("Camera:" in s["scene"] for s in ser)
    assert {a for *_, a in SERIES_SHOTS} == {True, False}
    assert build_series({"where": "on a pier"}, 8, seed=1) == build_series({"where": "on a pier"}, 8, seed=1)
    pa = photo_prompt("x", 2, look="phone", natural_skin=True, candid=True, anchor=True)
    assert "Image 3 is an earlier photo" in pa and NATURAL_SKIN in pa and CANDID in pa and "smartphone" in pa
    print(photo_prompt(a[0]["scene"]))
    print("ok")
