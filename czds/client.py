#!/usr/bin/env python3
# ICANN API for the Centralized Zones Data Service
# Developed by acidvegas (https://git.acid.vegas/czds)
# czds/client.py

import asyncio
import csv
import io
import json
import logging
import os

try:
    import aiohttp
except ImportError:
    raise ImportError("missing aiohttp library (pip install aiohttp)")

try:
    import aiofiles
except ImportError:
    raise ImportError("missing aiofiles library (pip install aiofiles)")

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("missing tqdm library (pip install tqdm)")

from .utils import humanize_bytes


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


# ---------------------------------------------------------------------------
# CZDS Client
# ---------------------------------------------------------------------------


class CZDS:
    """Client for the ICANN Centralized Zones Data Service (CZDS)."""

    def __init__(self, username: str, password: str):
        """
        Initialize the CZDS client.

        :param username: ICANN username
        :param password: ICANN password
        """

        self.username = username
        self.password = password
        self.headers = None

        connector = aiohttp.TCPConnector(
            keepalive_timeout=300,
            force_close=False,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=None, connect=60, sock_connect=60, sock_read=None
            ),
            headers={"Connection": "keep-alive"},
            raise_for_status=True,
        )

        logging.info("Initialized CZDS client")

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------

    async def __aenter__(self):
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the underlying HTTP session."""
        if self.session:
            await self.session.close()
            logging.debug("Closed aiohttp session")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self) -> str:
        """Authenticate with ICANN and return the access token."""

        logging.info("Authenticating with ICANN API...")

        payload = {"username": self.username, "password": self.password}

        async with self.session.post(
            "https://account-api.icann.org/api/authenticate", json=payload
        ) as response:

            result = await response.json()

            self.headers = {"Authorization": f'Bearer {result["accessToken"]}'}

            logging.info("Successfully authenticated with ICANN API")
            return result["accessToken"]

    # ------------------------------------------------------------------
    # Metadata / reports
    # ------------------------------------------------------------------

    async def fetch_zone_links(self) -> list:
        """Fetch all available CZDS zone download URLs."""

        logging.info("Fetching zone file links...")

        async with self.session.get(
            "https://czds-api.icann.org/czds/downloads/links", headers=self.headers
        ) as response:

            links = await response.json()

            logging.info(f"Successfully fetched {len(links):,} zone links")
            return links

    async def get_report(self, filepath: str = None, format: str = "csv") -> str | dict:
        """
        Download and scrub the zone request report.

        :param filepath: Optional output path
        :param format: 'csv' or 'json'
        """

        logging.info("Downloading zone stats report")

        async with self.session.get(
            "https://czds-api.icann.org/czds/requests/report", headers=self.headers
        ) as response:

            content = await response.text()

        # Scrub username
        content = content.replace(self.username, "nobody@no.name")

        if format.lower() == "json":
            reader = csv.DictReader(io.StringIO(content))
            content = json.dumps(
                [
                    {key.lower().replace(" ", "_"): value for key, value in row.items()}
                    for row in reader
                ],
                indent=4,
            )

        if filepath:
            async with aiofiles.open(filepath, "w") as f:
                await f.write(content)
            logging.info(f"Saved report to {filepath}")

        return content

    # ------------------------------------------------------------------
    # Zone downloads
    # ------------------------------------------------------------------

    async def download_zone(
        self, url: str, output_directory: str, semaphore: asyncio.Semaphore
    ):
        """
        Download a single zone file (.gz preserved as-is).

        :param url: Download URL
        :param output_directory: Destination directory
        :param semaphore: Concurrency limiter
        """

        async def _download():
            tld_name = url.split("/")[-1].split(".")[0]
            max_retries = 20
            retry_delay = 5

            headers = {
                **self.headers,
                "Connection": "keep-alive",
                "Keep-Alive": "timeout=600",
                "Accept-Encoding": "gzip",
            }

            for attempt in range(max_retries):
                try:
                    logging.info(
                        f"Starting download of {tld_name}"
                        + (f" (attempt {attempt + 1})" if attempt else "")
                    )

                    async with self.session.get(url, headers=headers) as response:

                        expected_size = int(response.headers.get("Content-Length", 0))
                        content_disp = response.headers.get("Content-Disposition")

                        if not expected_size:
                            raise ValueError("Missing Content-Length header")

                        if not content_disp:
                            raise ValueError("Missing Content-Disposition header")

                        filename = content_disp.split("filename=")[-1].strip('"')
                        filepath = os.path.join(output_directory, filename)

                        total_size = 0

                        with tqdm(
                            total=expected_size,
                            unit="B",
                            unit_scale=True,
                            desc=f"Downloading {tld_name}",
                            leave=False,
                        ) as pbar:

                            async with aiofiles.open(filepath, "wb") as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)
                                    total_size += len(chunk)
                                    pbar.update(len(chunk))

                        if total_size != expected_size:
                            os.remove(filepath)
                            raise IOError(
                                f"Incomplete download: "
                                f"{humanize_bytes(total_size)} / "
                                f"{humanize_bytes(expected_size)}"
                            )

                        logging.info(
                            f"Successfully downloaded {tld_name} "
                            f"({humanize_bytes(total_size)})"
                        )

                        return filepath

                except Exception as e:
                    if attempt + 1 >= max_retries:
                        logging.error(f"Failed to download {tld_name}: {e}")
                        if "filepath" in locals() and os.path.exists(filepath):
                            os.remove(filepath)
                        raise

                    logging.warning(
                        f"Download attempt {attempt + 1} failed for " f"{tld_name}: {e}"
                    )
                    await asyncio.sleep(retry_delay)

        async with semaphore:
            return await _download()

    async def download_zones(self, output_directory: str, concurrency: int):
        """
        Download all available zone files concurrently.

        :param output_directory: Destination directory
        :param concurrency: Max parallel downloads
        """

        os.makedirs(output_directory, exist_ok=True)

        zone_links = await self.fetch_zone_links()
        zone_links.sort()

        semaphore = asyncio.Semaphore(concurrency)

        logging.info(f"Downloading {len(zone_links):,} zone files...")

        tasks = [
            self.download_zone(url, output_directory, semaphore) for url in zone_links
        ]

        await asyncio.gather(*tasks)

        logging.info(f"Completed downloading {len(zone_links):,} zone files")
