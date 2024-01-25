from __future__ import annotations

import logging
import os
from getpass import getuser
from typing import Any

import requests
from pydantic import BaseModel

from .slurm_rest import (JobResponseProperties, JobsResponse, JobSubmission,
                         JobSubmissionResponse)

_SLURM_VERSION = "v0.0.38"


def get_slurm_token() -> str:
    return os.environ["SLURM_JWT"].strip()


class SlurmClient:
    def __init__(
        self,
        url: str,
        user_name: str | None = None,
        user_token: str | None = None,
    ):
        """Slurm client that communicates to Slurm via its RESTful API"""
        self._slurm_endpoint_url = f"{url}/slurm/{_SLURM_VERSION}"
        self._session = requests.Session()

        self.user = user_name if user_name else getuser()
        self.token = user_token if user_token else get_slurm_token()
        self._session.headers["X-SLURM-USER-NAME"] = self.user
        self._session.headers["X-SLURM-USER-TOKEN"] = self.token
        self._session.headers["Content-Type"] = "application/json"

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        return self._session.get(
            f"{self._slurm_endpoint_url}/{endpoint}", params=params, timeout=timeout
        )

    def _post(self, endpoint: str, data: BaseModel) -> requests.Response:
        return self._session.post(
            f"{self._slurm_endpoint_url}/{endpoint}",
            data.model_dump_json(exclude_defaults=True),
        )

    def _delete(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        return self._session.delete(
            f"{self._slurm_endpoint_url}/{endpoint}", params=params, timeout=timeout
        )

    def _get_response_json(self, response: requests.Response) -> dict:
        response.raise_for_status()
        try:
            return response.json()
        except:
            logging.error("Response not json: %s", response.content, exc_info=True)
            raise

    def get_jobs_response(self, job_id: int | None = None) -> JobsResponse:
        endpoint = f"job/{job_id}" if job_id is not None else "jobs"
        response = self._get(endpoint)
        return JobsResponse.model_validate(self._get_response_json(response))

    def get_job(self, job_id: int) -> JobResponseProperties:
        ji = self.get_jobs_response(job_id)
        if ji.jobs:
            n = len(ji.jobs)
            if n == 1:
                return ji.jobs[0]
            if n > 1:
                raise ValueError("Multiple jobs returned {ji.jobs}")
        raise ValueError("No job info found for job id {job_id}")

    def submit_job(self, job_submission: JobSubmission) -> JobSubmissionResponse:
        response = self._post("job/submit", job_submission)
        return JobSubmissionResponse.model_validate(self._get_response_json(response))

    def cancel_job(self, job_id: int) -> JobsResponse:
        response = self._delete(f"job/{job_id}")
        return JobsResponse.model_validate(self._get_response_json(response))
