# Single-cell Kaggle-ready PDF downloader
# Edit the three variables below as needed, then run this single cell.
#
# Input file format (urls.txt):
# - one URL per line
# - ignore blank lines and lines starting with #
# - optional title after the URL using either:
#     URL  Title : Spherical Vision
#   or
#     URL <whitespace> Spherical Vision
#
# Example:
# http://arxiv.org/pdf/2508.20221v1  Title : Spherical Vision

import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote
from tqdm.auto import tqdm
import math
import time
import os
# ------------------- USER CONFIG -------------------
input_file = "paper_links.txt"   # path to urls.txt in your Kaggle working directory
dest_dir = "./papers"     # destination folder (will be created)
workers = 8               # number of parallel downloads
timeout = 30              # per-request timeout (seconds)
# ---------------------------------------------------

# helpers
INVALID_FILENAME_CHARS = r'<>:"/\\|?*\0'
_filename_strip_re = re.compile(r'[%s]+' % re.escape(INVALID_FILENAME_CHARS))
_title_parse_re = re.compile(r'^(?P<url>\S+)(?:\s+(?:Title\s*:\s*)?(?P<title>.+))?$', re.IGNORECASE)

def sanitize_filename(name: str, max_len: int = 200) -> str:
    name = name.strip()
    name = _filename_strip_re.sub('_', name)
    # collapse spaces
    name = re.sub(r'\s+', ' ', name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name

def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    if path:
        base = os.path.basename(path)
        if base:
            return sanitize_filename(base)
    # fallback to host+timestamp
    safe_host = sanitize_filename(parsed.hostname or "file")
    return f"{safe_host}_{int(time.time())}.pdf"

def ensure_unique(dest_folder: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".pdf"
    candidate = f"{base}{ext}"
    i = 1
    while os.path.exists(os.path.join(dest_folder, candidate)):
        candidate = f"{base}({i}){ext}"
        i += 1
    return candidate

def parse_input_line(line: str):
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    m = _title_parse_re.match(line)
    if not m:
        return None
    url = m.group('url')
    title = m.group('title')
    if title:
        # strip possible "Title :" prefix inside capture, handled above, but be safe
        title = re.sub(r'^\s*Title\s*:\s*', '', title, flags=re.IGNORECASE).strip()
    return url, title

def download_one(session: requests.Session, url: str, title: str, dest_folder: str):
    try:
        with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
            resp.raise_for_status()
            # determine filename
            if title:
                name = title
                # if title already has .pdf or other extension, keep it but prefer .pdf
                if not os.path.splitext(name)[1]:
                    name = name + ".pdf"
                elif os.path.splitext(name)[1].lower() != ".pdf":
                    # convert extension to .pdf (common case: user included .pdf or not)
                    name = os.path.splitext(name)[0] + ".pdf"
                filename = sanitize_filename(name)
            else:
                # try Content-Disposition header
                cd = resp.headers.get("content-disposition", "")
                fname = None
                if cd:
                    # look for filename="..."
                    m = re.search(r'filename\*=.*\'\'(?P<n>[^;]+)', cd)
                    if m:
                        fname = m.group('n')
                    else:
                        m2 = re.search(r'filename="?([^";]+)"?', cd)
                        if m2:
                            fname = m2.group(1)
                if fname:
                    filename = sanitize_filename(fname)
                else:
                    filename = filename_from_url(resp.url or url)
                    if not filename.lower().endswith(".pdf"):
                        filename = filename + ".pdf"

            # ensure unique
            filename = ensure_unique(dest_folder, filename)
            fullpath = os.path.join(dest_folder, filename)

            # write stream to file with chunked progress
            total = resp.headers.get("content-length")
            if total and total.isdigit():
                total = int(total)
            else:
                total = None

            chunk_size = 1024 * 32
            with open(fullpath, "wb") as f:
                if total:
                    # show inner progress bar for large files
                    with tqdm(total=total, unit="B", unit_scale=True, desc=filename, leave=False) as pbar:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # unknown size: write without inner progress
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            return {"url": url, "path": fullpath, "size": os.path.getsize(fullpath), "ok": True}
    except Exception as e:
        return {"url": url, "error": str(e), "ok": False}

def main():
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    os.makedirs(dest_dir, exist_ok=True)

    tasks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for raw in f:
            parsed = parse_input_line(raw)
            if parsed:
                tasks.append(parsed)

    if not tasks:
        print("No URLs found in input. Make sure file has lines with URLs.")
        return

    # setup requests session with reasonable headers
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) KaggleDownloader/1.0",
        "Accept": "application/pdf,application/octet-stream,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    })

    results = []
    print(f"Starting downloads: {len(tasks)} files -> '{dest_dir}' using {workers} workers\n")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_task = {
            ex.submit(download_one, session, url, title, dest_dir): (url, title)
            for url, title in tasks
        }

        # Show a high-level progress bar for files
        for fut in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Files", unit="file"):
            res = fut.result()
            results.append(res)
            if not res.get("ok"):
                print("Failed:", res.get("url"), "->", res.get("error"))

    # summary
    succ = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]

    print("\nDownload summary:")
    print(f"  Success: {len(succ)}")
    if succ:
        # show top few files
        for r in succ[:10]:
            size_kb = r['size'] / 1024 if r.get('size') else 0
            print(f"    - {os.path.basename(r['path'])} ({size_kb:.1f} KB)")

    print(f"  Failed: {len(fail)}")
    if fail:
        for r in fail[:10]:
            print(f"    - {r.get('url')} -> {r.get('error')}")

    print(f"\nSaved to: {os.path.abspath(dest_dir)}")

if __name__ == "__main__":
    main()