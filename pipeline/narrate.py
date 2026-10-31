import argparse
import asyncio
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_map import pause_ms, prosody  # noqa: E402
from voice_select import assign_voices, build_pools, fetch_pools  # noqa: E402

CONCURRENCY = 4
RETRIES = 3


async def synth(job, path):
    import edge_tts
    part = path.with_suffix(".part")
    await edge_tts.Communicate(job["text"], job["voice"], rate=job["rate"],
                               pitch=job["pitch"], volume=job["volume"]).save(str(part))
    part.replace(path)


async def synth_retry(sem, job, path):
    async with sem:
        last = None
        for attempt in range(RETRIES):
            try:
                await synth(job, path)
                return
            except Exception as e:
                last = e
                await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"speech synthesis failed for unit {job['index']}: {last}")


def plan(labels, voices, flat=False):
    jobs = []
    for i, rec in enumerate(labels):
        if flat:
            voice, rate, pitch, volume = voices["narrator"], "+0%", "+0Hz", "+0%"
        else:
            voice = voices[rec["speaker"]]
            rate, pitch, volume = prosody(rec)
        text = rec["text"].strip()
        key = hashlib.sha1("|".join([text, voice, rate, pitch, volume]).encode("utf-8")).hexdigest()[:20]
        jobs.append({"index": i, "text": text, "voice": voice, "rate": rate,
                     "pitch": pitch, "volume": volume, "key": key})
    return jobs


async def synthesize_all(jobs, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = {j["key"]: j for j in jobs if not (cache_dir / f"{j['key']}.mp3").exists()}
    print(f"  {len(jobs) - len(todo)} cached, {len(todo)} to synthesize")
    sem = asyncio.Semaphore(CONCURRENCY)
    await asyncio.gather(*(synth_retry(sem, j, cache_dir / f"{k}.mp3") for k, j in todo.items()))


async def get_pools():
    try:
        male, female = await fetch_pools()
        if male and female:
            return build_pools(male, female)
    except Exception as e:
        print(f"  could not fetch the voice list ({type(e).__name__}); using built-in voices")
    return build_pools()


def assemble(labels, jobs, cache_dir, out_path, flat):
    from pydub import AudioSegment
    audio, manifest, prev = AudioSegment.empty(), [], None
    for rec, job in zip(labels, jobs):
        gap = 0 if prev is None else (200 if flat else pause_ms(prev, rec))
        if gap:
            audio += AudioSegment.silent(duration=gap)
        segment = AudioSegment.from_file(str(cache_dir / f"{job['key']}.mp3"), format="mp3")
        start = len(audio)
        audio += segment
        manifest.append({"index": job["index"], "text": rec["text"], "kind": rec["kind"],
                         "speaker": rec["speaker"], "voice": job["voice"], "emotion": rec["emotion"],
                         "intensity": rec["intensity"], "pace": rec["pace"], "rate": job["rate"],
                         "pitch": job["pitch"], "volume": job["volume"],
                         "start_ms": start, "end_ms": len(audio)})
        prev = rec
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(str(out_path), format="mp3", bitrate="128k")
    return manifest, len(audio)


async def _narrate(labels, out_path, cache_dir, flat):
    narrator, male, female = await get_pools()
    voices, genders = assign_voices(labels, narrator, male, female)
    jobs = plan(labels, voices, flat)
    await synthesize_all(jobs, cache_dir)
    return jobs, voices, genders


def narrate(labels, out_path, cache_dir="cache_audio", flat=False):
    out_path, cache_dir = Path(out_path), Path(cache_dir)
    with ThreadPoolExecutor(1) as pool:
        jobs, voices, genders = pool.submit(asyncio.run, _narrate(labels, out_path, cache_dir, flat)).result()
    manifest, length = assemble(labels, jobs, cache_dir, out_path, flat)
    if flat:
        voices, genders = {"narrator": voices["narrator"]}, {}
    info = {"audio": out_path.name, "duration_sec": round(length / 1000, 1), "flat": flat,
            "voices": voices, "speaker_gender": genders, "units": manifest}
    out_path.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {out_path} ({info['duration_sec']} s, {len(voices) - 1} character voices)")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out", default="output/audiobook.mp3")
    ap.add_argument("--cache", default="cache_audio")
    ap.add_argument("--flat", action="store_true")
    args = ap.parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    narrate(labels, args.out, args.cache, args.flat)


if __name__ == "__main__":
    main()
