"""Realism options for `video` and `story` (photos have their own wording in photos.py).

Options (job params): look (pro | phone | candid | portrait | film), natural_skin, candid, grain,
quality "best" (the avatar keyframe uses the full 40-step edit model; Wan keeps Lightning), and for videos
shot (stream | pov): how the clip is filmed and lit.
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
# How a video is filmed and lit (job param `shot`, videos only; one or none). "pov" names a handheld camera,
# so models.natural_prompts leaves its tripod sentence out; "stream" names no camera move and keeps it.
# "pov" is the look of phone footage, not a phone in the picture: besides the wording, the finished frames are
# softened, shaken and grained (PhoneLook below), so the look doesn't depend on the model following the text.
SHOTS = {
    "stream": ("Livestream webcam footage: a fixed webcam at eye level that never moves, the person sits facing "
               "the camera and talks to it, lit from the front by a ring light, soft even light on the face with "
               "round ring-light reflections in the eyes, a room behind them."),
    "pov": ("Amateur video shot on a handheld smartphone, seen from the point of view of the person filming: shaky "
            "handheld framing with small jolts, a casual home-video look, a soft low-resolution image with visible "
            "grain and compression, the phone's camera light as the main light, bright on what is close and falling "
            "off into a darker background. The person filming is never visible and no phone is visible in the "
            "picture."),
}
NATURAL_SKIN = "Natural skin with visible pores and fine texture, no retouching, no beauty filter."
CANDID = "Unposed, natural movement; people do not look at the camera."
GRAIN_STRENGTH = 3.0  # luminance noise sigma, in 0-255 levels


def style_text(params):
    """The sentences to append to a video/story prompt (and keyframe scene); '' with no options."""
    parts = [SHOTS.get(params.get("shot") or "", ""), VIDEO_LOOKS[params.get("look") or "pro"]]
    if params.get("candid"):
        parts.append(CANDID)
    if params.get("natural_skin"):
        parts.append(NATURAL_SKIN)
    return " ".join(p for p in parts if p)


def with_style(prompt, params):
    extra = style_text(params)
    return f"{prompt} {extra}" if extra else prompt


def frame_effect(params, seed):
    """What happens to each finished frame: a PhoneLook for the phone shot (it has its own grain), else True
    for the plain grain switch, else False."""
    if params.get("shot") == "pov":
        return PhoneLook(seed or 0)
    return bool(params.get("grain"))


class PhoneLook:
    """Finished frames made to look shot on a handheld phone: a slow hand shake (the frame is cropped by a
    few pixels and the crop wanders), a soft low-resolution image (scaled down and back up), and sensor
    grain. Call it on each uint8 HxWx3 frame in order; the same seed gives the same clip."""

    def __init__(self, seed, scale=0.55, shake=6, grain=7.0):
        import random

        self.seed, self.scale, self.shake, self.grain = seed, scale, shake, grain
        rnd = random.Random(seed)
        # Three slow sines per axis (0.1-0.45 Hz at 24 fps): sways like a hand, never jumps.
        self._waves = [[(rnd.uniform(0.2, 0.5), rnd.uniform(0.03, 0.12), rnd.uniform(0, 6.283)) for _ in range(3)]
                       for _ in range(2)]
        self._frame = 0
        self._rng = None

    def offset(self, i):
        """(dx, dy) of frame i in pixels, within +-shake."""
        import math

        out = []
        for waves in self._waves:
            total = sum(w for w, _, _ in waves)
            out.append(int(round(self.shake * sum(w * math.sin(f * i + p) for w, f, p in waves) / total)))
        return tuple(out)

    def __call__(self, frame):
        import numpy as np
        from PIL import Image

        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        h, w = frame.shape[:2]
        m = min(self.shake, (min(h, w) - 2) // 4)
        dx, dy = (max(-m, min(m, v)) for v in self.offset(self._frame))
        self._frame += 1
        crop = frame[m + dy:h - m + dy, m + dx:w - m + dx]
        scale = self.scale if w > 800 else max(self.scale, 0.75)  # a 640-wide frame is already low resolution
        small = (max(2, int(w * scale)), max(2, int(h * scale)))
        soft = Image.fromarray(np.ascontiguousarray(crop)).resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)
        noise = self._rng.normal(0.0, self.grain, (h, w)).astype(np.float32)
        return np.clip(np.asarray(soft).astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)


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
