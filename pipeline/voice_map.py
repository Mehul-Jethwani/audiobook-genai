EMOTION_STYLE = {
    "joy": (8, 6, 5),
    "warmth": (-4, 2, 0),
    "grief": (-14, -6, -8),
    "tension": (6, -2, 0),
    "anger": (8, -4, 10),
    "wonder": (-2, 6, 0),
    "reflection": (-10, -3, -5),
    "urgency": (16, 4, 8),
    "neutral": (0, 0, 0),
}
INTENSITY_SCALE = {"low": 0.6, "medium": 1.0, "high": 1.4}
PACE_RATE = {"slow": -8, "medium": 0, "fast": 8}


def clamp(value, low, high):
    return max(low, min(high, value))


def prosody(label):
    rate, pitch, volume = EMOTION_STYLE[label["emotion"]]
    scale = INTENSITY_SCALE[label["intensity"]]
    rate = clamp(round(rate * scale) + PACE_RATE[label["pace"]], -40, 40)
    pitch = clamp(round(pitch * scale), -20, 20)
    volume = clamp(round(volume * scale), -30, 30)
    return f"{rate:+d}%", f"{pitch:+d}Hz", f"{volume:+d}%"


def continues(prev):
    text = prev["text"].rstrip().rstrip("\"'\u201d\u2019").rstrip()
    return text.endswith((",", ":", ";", "\u2014"))


def pause_ms(prev, cur):
    if prev is None:
        return 0
    if continues(prev):
        return 120
    ms = 200
    if prev["speaker"] != cur["speaker"]:
        ms += 150
    if cur.get("tone_shift"):
        ms += 100
    return min(ms, 600)
