from __future__ import annotations

import logging
import os
from getpass import getuser
from typing import Any

import requests
from pydantic import BaseModel

from .slurm_rest import (
    JobInfo,
    JobSubmitResponseMsg,
    JobSubmitReq,
    OpenapiJobInfoResp,
    OpenapiJobSubmitResponse,
    OpenapiResp,
)

_SLURM_VERSION = "v0.0.40"


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
            json=data.model_dump(exclude_defaults=True),
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
        try:
            return response.json()
        except:
            logging.error("Response not json: %s", response.content, exc_info=True)
            raise

    def _has_openapi_errors(
        self,
        heading: str,
        oar: OpenapiResp | OpenapiJobInfoResp | OpenapiJobSubmitResponse,
    ) -> bool:
        if oar.warnings and oar.warnings.root:
            logging.warning(heading)
            for w in oar.warnings.root:
                logging.warning("    : %s", w)

        has_errors = oar.errors is not None and len(oar.errors.root) > 0
        if has_errors:
            logging.error(heading)
            assert oar.errors
            for e in oar.errors.root:
                logging.error("    : %s", e)

        return has_errors

    def get_job_response(self, job_id: int | None = None) -> list[JobInfo]:
        endpoint = f"job/{job_id}" if job_id is not None else "jobs"
        response = self._get(endpoint)
        ojir = OpenapiJobInfoResp.model_validate(self._get_response_json(response))
        if self._has_openapi_errors(f"Job query {job_id}:", ojir):
            return []
        return ojir.jobs.root

    def get_job(self, job_id: int) -> JobInfo:
        jobs = self.get_job_response(job_id)
        if jobs:
            n = len(jobs)
            if n == 1:
                return jobs[0]
            if n > 1:
                raise ValueError(f"Multiple jobs returned {jobs}")
        raise ValueError(f"No job info found for job id {job_id}")

    def submit_job(self, job_submission: JobSubmitReq) -> JobSubmitResponseMsg:
        response = self._post("job/submit", job_submission)
        if not response.ok:
            logging.error(job_submission.model_dump(exclude_defaults=True))
        ojsr = OpenapiJobSubmitResponse.model_validate(
            self._get_response_json(response)
        )
        self._has_openapi_errors(
            f"Job submit {ojsr.result.job_id if ojsr.result else 'None'}:", ojsr
        )
        response.raise_for_status()
        assert ojsr.result
        return ojsr.result

    def cancel_job(self, job_id: int) -> bool:
        response = self._delete(f"job/{job_id}")
        oar = OpenapiResp.model_validate(self._get_response_json(response))
        return not self._has_openapi_errors(f"Job query {job_id}:", oar)
