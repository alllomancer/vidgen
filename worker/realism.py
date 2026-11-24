"""Realism options for `video` and `story` (photos have their own wording in photos.py).

Options (job params): look (pro | phone | candid | portrait | film), natural_skin, candid, grain,
quality "best" (the avatar keyframe uses the full 40-step edit model; Wan keeps Lightning).
With no options the prompts are unchanged. numpy only, so it can be checked anywhere:
    python3 worker/realism.py
"""

LOOKS = ("pro", "phone", "candid", "portrait", "film")

# "pro" adds nothing: today's prompts.
VIDEO_LOOKS = {
    "pro": "",
    "phone": ("Handheld smartphone video, natural available light, slight natural camera shake, "
              "true-to-life colors, realistic textures."),
    "candid": ("Documentary-style footage on a 35mm lens, available light, unposed natural movement, "
               "true-to-life colors, realistic textures."),
    "portrait": ("Cinematic shot on an 85mm lens, shallow depth of field with soft background blur, "
                 "soft natural light, true-to-life skin tones."),
    "film": ("Shot on 35mm film, Kodak Portra look, gentle film grain, slightly muted warm colors, "
             "organic tones."),
}
NATURAL_SKIN = "Natural skin with visible pores and fine texture, no retouching, no beauty filter."
CANDID = "Unposed, natural movement; people do not look at the camera."
GRAIN_STRENGTH = 3.0  # luminance noise sigma, in 0-255 levels


def style_text(params):
    """The sentences to append to a video/story prompt (and keyframe scene); '' with no options."""
    parts = [VIDEO_LOOKS[params.get("look") or "pro"]]
    if params.get("candid"):
        parts.append(CANDID)
    if params.get("natural_skin"):
        parts.append(NATURAL_SKIN)
    return " ".join(p for p in parts if p)


def with_style(prompt, params):
    extra = style_text(params)
    return f"{prompt} {extra}" if extra else prompt


class Grain:
    """Film grain for video frames: a new seeded noise field per frame, like real grain.
    Call it on each uint8 HxWx3 frame in order; the same seed gives the same clip."""

    def __init__(self, seed, strength=GRAIN_STRENGTH):
        import numpy as np

        self.np = np
        self.rng = np.random.default_rng(seed)
        self.strength = strength

    def __call__(self, frame):
        np = self.np
        noise = self.rng.normal(0.0, self.strength, frame.shape[:2]).astype(np.float32)
        return np.clip(frame.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)


if __name__ == "__main__":
    assert with_style("a fox", {}) == "a fox" and with_style("a fox", {"look": "pro"}) == "a fox"
    s = with_style("a fox", {"look": "film", "candid": True, "natural_skin": True})
    assert s.startswith("a fox Shot on 35mm film") and CANDID in s and NATURAL_SKIN in s
    assert set(VIDEO_LOOKS) == set(LOOKS)
    try:
        import numpy as np
    except ImportError:
        print("ok (numpy not installed: grain not checked)")
    else:
        f = np.full((4, 4, 3), 128, np.uint8)
        a, b = Grain(1), Grain(1)
        first = a(f)
        assert (first == b(f)).all(), "same seed, same grain"
        assert not (first == a(f)).all(), "each frame gets new grain"
        print("ok")
