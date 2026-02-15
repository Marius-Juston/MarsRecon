# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""MarsHiRISE dataset."""
import asyncio
import json
import logging
import logging.config
import multiprocessing
import os.path
import pathlib
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import aiohttp
import pdr
from aiohttp import ClientResponseError, ClientConnectorError
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import (
    Path,
    Sample,
    download_url
)

logger = logging.getLogger(__name__)

CONFIG = 'logger_config.json'


def filter_maker(level):
    level = getattr(logging, level)

    def filter(record):
        return record.levelno <= level

    return filter



async def download_file(session, url, path, max_retries=8, base_delay=1.0, max_delay=60.0):
    """
    Downloads a file with an Exponential Backoff and Jitter control loop
    to handle 503 and 429 server saturation errors.
    """
    if path.exists():
        logger.warning(f"File {path} already exists. Skipping download.")
        return

    attempt = 0
    while attempt <= max_retries:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)

                chunk_size = 1 << 20

                with open(path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        await asyncio.to_thread(f.write, chunk)

            # If successful, break the loop and return
            return

        except ClientResponseError as e:
            # 429: Too Many Requests, 503: Service Unavailable, 504: Gateway Timeout
            if e.status in {429, 503, 504}:
                logger.warning(f"Server saturated ({e.status}) for {url}. Attempt {attempt + 1}/{max_retries}.")
            else:
                # Fatal error (e.g., 404 Not Found, 403 Forbidden). Do not retry.
                logger.error(f"Fatal HTTP {e.status} for {url}: {e.message}")
                return

        except (ClientConnectorError, asyncio.TimeoutError) as e:
            # Handle socket drops and timeouts which also occur during saturation
            logger.warning(f"Connection dropped for {url}. Attempt {attempt + 1}/{max_retries}. Error: {e}")

        except Exception as e:
            logger.error(f"Unexpected failure downloading {url}: {e}")
            return

        # Calculate Jittered Exponential Backoff
        attempt += 1
        if attempt <= max_retries:
            # T_n ~ U(0, min(t_base * 2^n, T_max))
            upper_bound = min(base_delay * (2 ** attempt), max_delay)
            sleep_time = random.uniform(0, upper_bound)

            logger.info(f"Backing off for {sleep_time:.2f} seconds before retrying {url}")
            await asyncio.sleep(sleep_time)
        else:
            logger.error(f"Max retries ({max_retries}) exhausted for {url}. File skipped.")


async def download_many(tasks, concurrency=None):
    if concurrency is None:
        concurrency = os.cpu_count()

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(
            *(download_file(session, url, path) for url, path in tasks)
        )


def worker_process(tasks, concurrency_per_process):
    asyncio.run(download_many(tasks, concurrency_per_process))


# We want to use
class MarsHiRISE(NonGeoDataset):
    """Mars HiRISE Experiment Data Records dataset.

    HiRISE <https://pds-imaging.jpl.nasa.gov/volumes/mro.html> is a large dataset containing high resolution
    images of the Mars surface. This dataset focuses specifically on the RDR products which are
    radiometrically-corrected images resampled to a standard map projection. They are formatted and organized
    according to the standards of the PDS.

    Further description of the dataset is located in <https://hirise-pds.lpl.arizona.edu/PDS/AAREADME.TXT>
    """

    def __getitem__(self, index: int) -> Sample:
        pass

    def __len__(self) -> int:
        pass

    url = 'https://hirise-pds.lpl.arizona.edu/PDS'

    rdr_name = 'RDRCUMINDEX'

    def __init__(
            self,
            root: Path = '/scratch/mars_hirise',
            split: str = 'train',
            transforms: Callable[[Sample], Sample] | None = None,
            download: bool = False,
            checksum: bool = False,
    ) -> None:
        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._cache_index_data = None

        self._verify()

    def _verify(self):
        self._download()

    def _download(self) -> None:
        self._download_index()

        self._read_index()

        self.file_list = self._get_file_list()

        self._download_images_high_speed()

    def _read_index(self):
        data = pdr.read(f"{self.rdr_name}.LBL")
        data.load('all')

        self._cache_index_data = data

    def _download_index(self) -> None:
        for path in [".LBL", ".TAB"]:
            full_name = self.rdr_name + path

            download_url(os.path.join(self.url, "INDEX", full_name), self.root, full_name)

    def _get_file_list(self) -> list[str]:
        return self._cache_index_data['RDR_INDEX_TABLE']["FILE_NAME_SPECIFICATION"]

    def build_tasks(self):
        tasks = []

        for rel in self.file_list:
            suffix = len('.JP2')

            raw = pathlib.Path(rel[:-suffix])

            for suffix in [".JP2", ".LBL"]:
                url = f"{self.url}/{raw}{suffix}"
                local = pathlib.Path(self.root) / "images" / f"{raw.name}{suffix}"
                tasks.append((url, local))

        return tasks

    def _download_images_high_speed(self, subdir='images') -> None:
        task_list = self.build_tasks()
        total_tasks = len(task_list)

        logger.info(f"Downloading {len(task_list)} images to {subdir}")

        if total_tasks == 0:
            return

        num_cores = multiprocessing.cpu_count()

        active_processes = min(num_cores, 64)
        concurrency_per_process = 8

        chunk_size = (total_tasks + active_processes - 1) // active_processes
        task_chunks = [
            task_list[i * chunk_size:(i + 1) * chunk_size]
            for i in range(active_processes)
        ]

        logger.info(f"Distributing payload across {active_processes} processes.")

        with ProcessPoolExecutor(max_workers=active_processes) as executor:
            futures = [
                executor.submit(worker_process, chunk, concurrency_per_process)
                for chunk in task_chunks if chunk
            ]

            for future in futures:
                future.result()


def setup_logging():
    with open(CONFIG, 'r') as f:
        config_dict = json.load(f)

    logging.config.dictConfig(config_dict)


def main():
    setup_logging()

    # # import urllib.request
    # # urllib.request.urlretrieve("https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.LBL", "RDRCUMINDEX.LBL")
    # # urllib.request.urlretrieve("https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB", "RDRCUMINDEX.TAB")
    #
    # data = pdr.read("RDRCUMINDEX.LBL")
    # data.load('all')
    #
    # print(data)
    #
    # get_data_column_index = "FILE_NAME_SPECIFICATION"
    # index_table = data['RDR_INDEX_TABLE']
    #
    # file_names = index_table["FILE_NAME_SPECIFICATION"]
    #
    # full_path = MarsHiRISE.url + file_names
    #
    # print(full_path)
    #
    # val = index_table['RATIONALE_DESC'].unique()
    # print(val)

    dataset = MarsHiRISE()


if __name__ == '__main__':
    main()
