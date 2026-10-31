from collections import Counter

NARRATOR_PREFERENCE = ["en-GB-RyanNeural", "en-US-GuyNeural", "en-US-ChristopherNeural"]
MALE_PREFERENCE = ["en-US-GuyNeural", "en-GB-ThomasNeural", "en-US-EricNeural", "en-AU-WilliamNeural",
                   "en-US-ChristopherNeural", "en-IN-PrabhatNeural", "en-GB-RyanNeural"]
FEMALE_PREFERENCE = ["en-US-JennyNeural", "en-GB-SoniaNeural", "en-US-AriaNeural", "en-AU-NatashaNeural",
                     "en-US-MichelleNeural", "en-IN-NeerjaNeural", "en-GB-LibbyNeural"]


async def fetch_pools():
    import edge_tts
    voices = await edge_tts.list_voices()
    male, female = [], []
    for v in voices:
        name = v["ShortName"]
        if not name.startswith("en-") or not name.endswith("Neural"):
            continue
        (male if v["Gender"] == "Male" else female).append(name)
    return sorted(male), sorted(female)


def build_pools(male_available=None, female_available=None):
    def order(preferred, available):
        if not available:
            return list(preferred)
        first = [v for v in preferred if v in available]
        return first + [v for v in available if v not in first]

    male = order(MALE_PREFERENCE, male_available)
    female = order(FEMALE_PREFERENCE, female_available)
    narrator = next((v for v in NARRATOR_PREFERENCE if v in male + female), male[0])
    male = [v for v in male if v != narrator] or male
    return narrator, male, female


MALE_TITLES = {"MR", "SIR", "LORD", "MASTER", "KING", "PRINCE", "FATHER", "BROTHER", "UNCLE", "MAN", "BOY"}
FEMALE_TITLES = {"MRS", "MISS", "MS", "LADY", "MADAME", "QUEEN", "PRINCESS", "MOTHER", "SISTER", "AUNT", "WOMAN", "GIRL"}
_detector = []


def name_gender(speaker):
    first = speaker.split()[0].rstrip(".")
    if first in MALE_TITLES:
        return "male"
    if first in FEMALE_TITLES:
        return "female"
    try:
        if not _detector:
            import gender_guesser.detector as gd
            _detector.append(gd.Detector(case_sensitive=False))
        guess = _detector[0].get_gender(first.title())
    except ImportError:
        return None
    return {"male": "male", "mostly_male": "male", "female": "female", "mostly_female": "female"}.get(guess)


def speaker_genders(labels):
    votes = {}
    for rec in labels:
        if rec["speaker"] == "narrator":
            continue
        votes.setdefault(rec["speaker"], Counter())[rec["gender"]] += 1
    result = {}
    for speaker, counter in votes.items():
        known = [(c, g) for g, c in counter.items() if g in ("male", "female")]
        result[speaker] = name_gender(speaker) or (max(known)[1] if known else "unknown")
    return result


def assign_voices(labels, narrator, male, female):
    genders = speaker_genders(labels)
    order = []
    for rec in labels:
        if rec["speaker"] != "narrator" and rec["speaker"] not in order:
            order.append(rec["speaker"])
    voices, used = {"narrator": narrator}, {"male": 0, "female": 0}
    flip = 0
    for speaker in order:
        gender = genders[speaker]
        if gender == "unknown":
            gender = ("male", "female")[flip % 2]
            flip += 1
        pool = male if gender == "male" else female
        voices[speaker] = pool[used[gender] % len(pool)]
        used[gender] += 1
    return voices, genders
