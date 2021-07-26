import os
from datetime import timedelta
from pathlib import Path
from job_scheduler import JobScheduler


class JobController:

    def __init__(self, working_directory, cluster_output_dir, project, priority, cpus=16, timeout=timedelta(minutes=2)):
        """JobController can be used for cluster job submission"""

        self.working_directory = Path(working_directory)
        self.cluster_output_dir = Path(cluster_output_dir)
        self.project = project
        self.priority = priority
        self.cpus = cpus
        self.timeout = timeout
        self.scheduler = None

    def run(self, data_chunker_class, data_chunker_args, data_to_split, processing_script):
        # split data
        data_chunker = data_chunker_class(*data_chunker_args)
        input_paths = data_chunker.chunk(self.working_directory, data_to_split)

        # run jobs
        self.scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project, self.priority,
                                      self.cpus, self.timeout)
        self.scheduler.run(processing_script, input_paths, log_path=None)

        # rerun any killed jobs
        self.rerun_killed_jobs(processing_script)

        # check all jobs completed successfully and run aggregation
        if self.scheduler.get_success():
            chunked_results = self.scheduler.get_output_paths()
            aggregated_data_path = data_chunker.aggregate(self.cluster_output_dir, chunked_results)
            return aggregated_data_path
        else:
            raise RuntimeError(f"All jobs were not successful. Aggregation not performed\n")

    def rerun_killed_jobs(self, processing_script):
        if not self.scheduler.get_success():
            job_history = self.scheduler.job_history
            failed_jobs = [job_info for job_info in job_history[0].values() if job_info["final_state"] != "SUCCESS"]

            if any(self.scheduler.job_completion_status.values()):
                killed_jobs = self.filter_killed_jobs(failed_jobs)
                killed_jobs_inputs = [job["input_path"] for job in killed_jobs]
                self.scheduler.resubmit_jobs(processing_script, killed_jobs_inputs)
            else:
                raise RuntimeError(f"All jobs failed\n")

    def filter_killed_jobs(self, jobs):
        killed_jobs = [job_info for job_info in jobs if job_info["term_sig"] == "SIGKILL"]
        return killed_jobs
