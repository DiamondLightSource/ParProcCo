# generated by datamodel-codegen:
#   filename:  slurmdb-rest.yaml
#   timestamp: 2023-08-18T16:27:32+00:00
# ruff: noqa: E501 # ignore long lines

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, RootModel


class Default(BaseModel):
    """
    Default settings
    """

    qos: str | None = None
    """
    Default QOS
    """


class Per(BaseModel):
    """
    Max jobs per settings
    """

    wall_clock: int | None = None
    """
    Max wallclock per job
    """


class Jobs(BaseModel):
    """
    Max jobs settings
    """

    per: Per | None = None
    """
    Max jobs per settings
    """


class Account(BaseModel):
    """
    Max per accounting settings
    """

    wall_clock: int | None = None
    """
    Max wallclock per account
    """


class Per1(BaseModel):
    """
    Max per settings
    """

    account: Account | None = None
    """
    Max per accounting settings
    """


class Min(BaseModel):
    """
    Min settings
    """

    priority_threshold: int | None = None
    """
    Min priority threshold
    """


class Usage(BaseModel):
    """
    Association usage
    """

    accrue_job_count: int | None = None
    """
    Jobs accuring priority
    """
    effective_normalized_usage: float | None = None
    """
    Effective normalized usage
    """
    fairshare_factor: float | None = None
    """
    Fairshare factor
    """
    fairshare_level: float | None = None
    """
    Fairshare level
    """
    fairshare_shares: int | None = None
    """
    Fairshare shares
    """
    group_used_wallclock: float | None = None
    """
    Group used wallclock time (s)
    """
    job_count: int | None = None
    """
    Total jobs submitted
    """
    normalized_priority: int | None = None
    """
    Currently active jobs
    """
    normalized_shares: float | None = None
    """
    Normalized shares
    """
    raw_usage: int | None = None
    """
    Raw usage
    """


class DbAssociationShortInfo(BaseModel):
    account: str | None = None
    """
    Account name
    """
    cluster: str | None = None
    """
    Cluster name
    """
    partition: str | None = None
    """
    Partition name (optional)
    """
    user: str | None = None
    """
    User name
    """


class Associations(BaseModel):
    """
    Information about associations
    """

    root: DbAssociationShortInfo | None = None


class Controller(BaseModel):
    """
    Information about controller
    """

    host: str | None = None
    """
    Hostname
    """
    port: int | None = None
    """
    Port number
    """


class DbCoordinatorInfo(BaseModel):
    direct: int | None = None
    """
    If user is coordinator of this account directly or coordinator status was inherited from a higher account in the tree
    """
    name: str | None = None
    """
    Name of user
    """


class Time(BaseModel):
    """
    Time values
    """

    average: int | None = None
    """
    Average time spent processing this RPC type
    """
    total: int | None = None
    """
    Total time spent processing this RPC type
    """


class RPC(BaseModel):
    """
    Statistics by RPC type
    """

    count: int | None = None
    """
    Number of RPCs
    """
    rpc: str | None = None
    """
    RPC type
    """
    time: Time | None = None
    """
    Time values
    """


class Rollup(BaseModel):
    """
    Rollup statistics
    """

    last_cycle: int | None = None
    """
    Timestamp of last cycle
    """
    last_run: int | None = None
    """
    Timestamp of last rollup
    """
    max_cycle: int | None = None
    """
    Max time of all cycles
    """
    mean_cycles: int | None = None
    """
    Average time (s) of cycle
    """
    total_time: int | None = None
    """
    Total time (s) spent doing rollup
    """
    type: str | None = None
    """
    Type of rollup
    """


class Time1(BaseModel):
    """
    Time values
    """

    average: int | None = None
    """
    Average time spent processing each user RPC
    """
    total: int | None = None
    """
    Total time spent processing each user RPC
    """


class User(BaseModel):
    """
    Statistics by user RPCs
    """

    count: int | None = None
    """
    Number of RPCs
    """
    time: Time1 | None = None
    """
    Time values
    """
    user: str | None = None
    """
    User name
    """


class Statistics(BaseModel):
    RPCs: list[RPC] | None = None
    rollups: list[Rollup] | None = None
    time_start: int | None = None
    """
    Unix timestamp of start time
    """
    users: list[User] | None = None


class DbError(BaseModel):
    description: str | None = None
    """
    Explaination of cause of error
    """
    error: str | None = None
    """
    Error message
    """
    error_number: int | None = None
    """
    Slurm internal error number
    """
    source: str | None = None
    """
    Where error occured in the source
    """


class DbErrors(RootModel):
    """
    Slurm errors
    """

    root: list[DbError]
    """
    Slurm errors
    """


class Running(BaseModel):
    """
    Limits on array settings
    """

    tasks: int | None = None
    """
    Max running tasks in array at any one time
    """


class Max1(BaseModel):
    """
    Limits on array settings
    """

    running: Running | None = None
    """
    Limits on array settings
    """


class Limits(BaseModel):
    """
    Limits on array settings
    """

    max: Max1 | None = None
    """
    Limits on array settings
    """


class Array(BaseModel):
    """
    Array properties (optional)
    """

    job_id: int | None = None
    """
    Job id of array
    """
    limits: Limits | None = None
    """
    Limits on array settings
    """
    task: str | None = None
    """
    Array task
    """
    task_id: int | None = None
    """
    Array task id
    """


class Comment(BaseModel):
    """
    Job comments by type
    """

    administrator: str | None = None
    """
    Administrator set comment
    """
    job: str | None = None
    """
    Job comment
    """
    system: str | None = None
    """
    System set comment
    """


class Het(BaseModel):
    """
    Heterogeneous Job details (optional)
    """

    job_id: int | None = None
    """
    Parent HetJob id
    """
    job_offset: int | None = None
    """
    Offset of this job to parent
    """


class Mcs(BaseModel):
    """
    Multi-Category Security
    """

    label: str | None = None
    """
    Assigned MCS label
    """


class Required(BaseModel):
    """
    Job run requirements
    """

    CPUs: int | None = None
    """
    Required number of CPUs
    """
    memory: int | None = None
    """
    Required amount of memory (MiB)
    """


class Reservation(BaseModel):
    """
    Reservation usage details
    """

    id: int | None = None
    """
    Database id of reservation
    """
    name: int | None = None
    """
    Name of reservation
    """


class State(BaseModel):
    """
    State properties of job
    """

    current: str | None = None
    """
    Current state of job
    """
    reason: str | None = None
    """
    Last reason job didn't run
    """


class System(BaseModel):
    """
    System time values
    """

    microseconds: int | None = None
    """
    Total number of CPU-seconds used by the system on behalf of the process (in kernel mode), in microseconds
    """
    seconds: int | None = None
    """
    Total number of CPU-seconds used by the system on behalf of the process (in kernel mode), in seconds
    """


class Total(BaseModel):
    """
    System time values
    """

    microseconds: int | None = None
    """
    Total number of CPU-seconds used by the job, in microseconds
    """
    seconds: int | None = None
    """
    Total number of CPU-seconds used by the job, in seconds
    """


class User1(BaseModel):
    """
    User land time values
    """

    microseconds: int | None = None
    """
    Total number of CPU-seconds used by the job in user land, in microseconds
    """
    seconds: int | None = None
    """
    Total number of CPU-seconds used by the job in user land, in seconds
    """


class Time2(BaseModel):
    """
    Time properties
    """

    elapsed: int | None = None
    """
    Total time elapsed
    """
    eligible: int | None = None
    """
    Total time eligible to run
    """
    end: int | None = None
    """
    Timestamp of when job ended
    """
    limit: int | None = None
    """
    Job wall clock time limit
    """
    start: int | None = None
    """
    Timestamp of when job started
    """
    submission: int | None = None
    """
    Timestamp of when job submitted
    """
    suspended: int | None = None
    """
    Timestamp of when job last suspended
    """
    system: System | None = None
    """
    System time values
    """
    total: Total | None = None
    """
    System time values
    """
    user: User1 | None = None
    """
    User land time values
    """


class Wckey(BaseModel):
    """
    Job assigned wckey details
    """

    flags: list[str] | None = None
    """
    wckey flags
    """
    wckey: str | None = None
    """
    Job assigned wckey
    """


class Signal(BaseModel):
    """
    Signal details (if signaled)
    """

    name: str | None = None
    """
    Name of signal received
    """
    signal_id: int | None = None
    """
    Signal number process received
    """


class DbJobExitCode(BaseModel):
    return_code: int | None = None
    """
    Return code from parent process
    """
    signal: Signal | None = None
    """
    Signal details (if signaled)
    """
    status: str | None = None
    """
    Job exit status
    """


class RequestedFrequency(BaseModel):
    """
    CPU frequency requested
    """

    max: int | None = None
    """
    Max CPU frequency
    """
    min: int | None = None
    """
    Min CPU frequency
    """


class CPU(BaseModel):
    """
    CPU properties
    """

    governor: list[str] | None = None
    """
    CPU governor
    """
    requested_frequency: RequestedFrequency | None = None
    """
    CPU frequency requested
    """


class Nodes(BaseModel):
    """
    Node details
    """

    count: int | None = None
    """
    Total number of nodes in step
    """
    range: str | None = None
    """
    Nodes in step
    """


class CPU1(BaseModel):
    """
    Statistics of CPU
    """

    actual_frequency: int | None = None
    """
    Actual frequency of CPU during step
    """


class Energy(BaseModel):
    """
    Statistics of energy
    """

    consumed: int | None = None
    """
    Energy consumed during step
    """


class Statistics1(BaseModel):
    """
    Statistics of job step
    """

    CPU: CPU1 | None = None
    """
    Statistics of CPU
    """
    energy: Energy | None = None
    """
    Statistics of energy
    """


class Het1(BaseModel):
    """
    Heterogeneous job details
    """

    component: int | None = None
    """
    Parent HetJob component id
    """


class Step(BaseModel):
    """
    Step details
    """

    het: Het1 | None = None
    """
    Heterogeneous job details
    """
    id: str | None = None
    """
    Step id
    """
    job_id: int | None = None
    """
    Parent job id
    """
    name: str | None = None
    """
    Step name
    """


class Tasks(BaseModel):
    """
    Task properties
    """

    count: int | None = None
    """
    Number of tasks in step
    """


class Time3(BaseModel):
    """
    Time properties
    """

    elapsed: int | None = None
    """
    Total time elapsed
    """
    end: int | None = None
    """
    Timestamp of when job ended
    """
    start: int | None = None
    """
    Timestamp of when job started
    """
    suspended: int | None = None
    """
    Timestamp of when job last suspended
    """
    system: System | None = None
    """
    System time values
    """
    total: Total | None = None
    """
    System time values
    """
    user: User1 | None = None
    """
    User land time values
    """


class Version(BaseModel):
    major: int | None = None
    micro: int | None = None
    minor: int | None = None


class Slurm(BaseModel):
    """
    Slurm information
    """

    release: str | None = None
    """
    version specifier
    """
    version: Version | None = None


class Plugin(BaseModel):
    name: str | None = None
    type: str | None = None


class DbMeta(BaseModel):
    slurm: Slurm | None = Field(None, validation_alias="Slurm")
    """
    Slurm information
    """
    plugin: Plugin | None = None


class Per4(BaseModel):
    """
    Max accuring priority per setting
    """

    account: int | None = None
    """
    Max accuring priority per account
    """
    user: int | None = None
    """
    Max accuring priority per user
    """


class Accruing(BaseModel):
    """
    Limits on accruing priority
    """

    per: Per4 | None = None
    """
    Max accuring priority per setting
    """


class Per5(BaseModel):
    """
    Limits on active jobs per settings
    """

    account: int | None = None
    """
    Max jobs per account
    """
    user: int | None = None
    """
    Max jobs per user
    """


class ActiveJobs(BaseModel):
    """
    Limits on active jobs settings
    """

    per: Per5 | None = None
    """
    Limits on active jobs per settings
    """


class Jobs1(BaseModel):
    """
    Limits on jobs settings
    """

    active_jobs: ActiveJobs | None = None
    """
    Limits on active jobs settings
    """


class Per8(BaseModel):
    """
    Limit on wallclock per settings
    """

    job: int | None = None
    """
    Max wallclock per job
    """
    qos: int | None = None
    """
    Max wallclock per QOS
    """


class WallClock(BaseModel):
    """
    Limit on wallclock settings
    """

    per: Per8 | None = None
    """
    Limit on wallclock per settings
    """


class Preempt(BaseModel):
    """
    Preemption settings
    """

    exempt_time: int | None = None
    """
    Grace period (s) before jobs can preempted
    """
    qos: list[str] | None = Field(None, validation_alias="list")
    """
    List of preemptable QOS
    """
    mode: list[str] | None = None
    """
    List of preemption modes
    """


class DbTresListItem(BaseModel):
    count: int | None = None
    """
    count of TRES
    """
    id: int | None = None
    """
    database id
    """
    name: str | None = None
    """
    TRES name (optional)
    """
    type: str | None = None
    """
    TRES type
    """


class DbTresList(RootModel):
    """
    TRES list of attributes
    """

    root: list[DbTresListItem]
    """
    TRES list of attributes
    """


class DbTresUpdate(BaseModel):
    tres: DbTresList | None = None


class Default1(BaseModel):
    """
    Default settings
    """

    account: str | None = None
    """
    Default account name
    """
    wckey: str | None = None
    """
    Default wckey
    """


class DbUser(BaseModel):
    """
    User description
    """

    administrator_level: str | None = None
    """
    Description of administrator level
    """
    associations: list[DbAssociationShortInfo] | None = None
    """
    Assigned associations
    """
    coordinators: list[DbCoordinatorInfo] | None = None
    """
    List of assigned coordinators
    """
    default: Default1 | None = None
    """
    Default settings
    """
    flags: list[str] | None = None
    """
    List of properties of user
    """
    name: str | None = None
    """
    User name
    """


class Meta(RootModel):
    root: Any


class DbAccount(BaseModel):
    """
    Account description
    """

    associations: list[DbAssociationShortInfo] | None = None
    """
    List of assigned associations
    """
    coordinators: list[DbCoordinatorInfo] | None = None
    """
    List of assigned coordinators
    """
    description: str | None = None
    """
    Description of account
    """
    flags: list[str] | None = None
    """
    List of properties of account
    """
    name: str | None = None
    """
    Name of account
    """
    organization: str | None = None
    """
    Assigned organization of account
    """


class DbAccountInfo(BaseModel):
    accounts: list[DbAccount] | None = None
    """
    List of accounts
    """
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbAccountResponse(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbAccounting(BaseModel):
    TRES: DbTresList | None = None
    allocated: int | None = None
    """
    total seconds allocated
    """
    id: int | None = None
    """
    association/wckey ID
    """
    start: int | None = None
    """
    UNIX timestamp when accounting period started
    """


class Per2(BaseModel):
    """
    Max TRES minutes per settings
    """

    job: DbTresList | None = None


class Minutes(BaseModel):
    """
    Max TRES minutes settings
    """

    per: Per2 | None = None
    """
    Max TRES minutes per settings
    """
    total: DbTresList | None = None


class Per3(BaseModel):
    """
    Max TRES per settings
    """

    job: DbTresList | None = None
    node: DbTresList | None = None


class Tres(BaseModel):
    """
    Max TRES settings
    """

    minutes: Minutes | None = None
    """
    Max TRES minutes settings
    """
    per: Per3 | None = None
    """
    Max TRES per settings
    """
    total: DbTresList | None = None


class Max(BaseModel):
    """
    Max settings
    """

    jobs: Jobs | None = None
    """
    Max jobs settings
    """
    per: Per1 | None = None
    """
    Max per settings
    """
    tres: Tres | None = None
    """
    Max TRES settings
    """


class DbAssociation(BaseModel):
    """
    Association description
    """

    QOS: list[str] | None = None
    """
    Assigned QOS
    """
    account: str | None = None
    """
    Assigned account
    """
    cluster: str | None = None
    """
    Assigned cluster
    """
    default: Default | None = None
    """
    Default settings
    """
    flags: list[str] | None = None
    """
    List of properties of association
    """
    max: Max | None = None
    """
    Max settings
    """
    min: Min | None = None
    """
    Min settings
    """
    parent_account: str | None = None
    """
    Parent account name
    """
    partition: str | None = None
    """
    Assigned partition
    """
    priority: int | None = None
    """
    Assigned priority
    """
    shares_raw: int | None = None
    """
    Raw fairshare shares
    """
    usage: Usage | None = None
    """
    Association usage
    """
    user: str | None = None
    """
    Assigned user
    """


class DbAssociationsInfo(BaseModel):
    associations: list[DbAssociation] | None = None
    """
    Array of associations
    """
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbConfigResponse(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbDiag(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None
    statistics: Statistics | None = None


class Tres1(BaseModel):
    """
    TRES settings
    """

    allocated: DbTresList | None = None
    requested: DbTresList | None = None


class Consumed(BaseModel):
    """
    TRES requested for job
    """

    average: DbTresList | None = None
    max: DbTresList | None = None
    min: DbTresList | None = None
    total: DbTresList | None = None


class Requested(BaseModel):
    """
    TRES requested for job
    """

    average: DbTresList | None = None
    max: DbTresList | None = None
    min: DbTresList | None = None
    total: DbTresList | None = None


class Tres2(BaseModel):
    """
    TRES usage
    """

    allocated: DbTresList | None = None
    consumed: Consumed | None = None
    """
    TRES requested for job
    """
    requested: Requested | None = None
    """
    TRES requested for job
    """


class DbJobStep(BaseModel):
    cpu: CPU | None = Field(None, validation_alias="CPU")
    """
    CPU properties
    """
    exit_code: DbJobExitCode | None = None
    kill_request_user: str | None = None
    """
    User who requested job killed
    """
    nodes: Nodes | None = None
    """
    Node details
    """
    pid: str | None = None
    """
    First process PID
    """
    state: str | None = None
    """
    State of job step
    """
    statistics: Statistics1 | None = None
    """
    Statistics of job step
    """
    step: Step | None = None
    """
    Step details
    """
    task: str | None = None
    """
    Task distribution properties
    """
    tasks: Tasks | None = None
    """
    Task properties
    """
    time: Time3 | None = None
    """
    Time properties
    """
    tres: Tres2 | None = None
    """
    TRES usage
    """


class Per6(BaseModel):
    """
    Max TRES minutes per settings
    """

    account: DbTresList | None = None
    job: DbTresList | None = None
    qos: DbTresList | None = None
    user: DbTresList | None = None


class Minutes1(BaseModel):
    """
    Max TRES minutes settings
    """

    per: Per6 | None = None
    """
    Max TRES minutes per settings
    """


class Per7(BaseModel):
    """
    Max TRES per settings
    """

    account: DbTresList | None = None
    job: DbTresList | None = None
    node: DbTresList | None = None
    user: DbTresList | None = None


class Tres3(BaseModel):
    """
    Limits on TRES
    """

    minutes: Minutes1 | None = None
    """
    Max TRES minutes settings
    """
    per: Per7 | None = None
    """
    Max TRES per settings
    """


class Max2(BaseModel):
    """
    Limits on max settings
    """

    accruing: Accruing | None = None
    """
    Limits on accruing priority
    """
    jobs: Jobs1 | None = None
    """
    Limits on jobs settings
    """
    tres: Tres3 | None = None
    """
    Limits on TRES
    """
    wall_clock: WallClock | None = None
    """
    Limit on wallclock settings
    """


class Per9(BaseModel):
    """
    Min tres per settings
    """

    job: DbTresList | None = None


class Tres4(BaseModel):
    """
    Min tres settings
    """

    per: Per9 | None = None
    """
    Min tres per settings
    """


class Min1(BaseModel):
    """
    Min limit settings
    """

    priority_threshold: int | None = None
    """
    Min priority threshold
    """
    tres: Tres4 | None = None
    """
    Min tres settings
    """


class Limits1(BaseModel):
    """
    Assigned limits
    """

    factor: float | None = None
    """
    factor to apply to TRES count for associations using this QOS
    """
    max: Max2 | None = None
    """
    Limits on max settings
    """
    min: Min1 | None = None
    """
    Min limit settings
    """


class DbQos(BaseModel):
    """
    QOS description
    """

    description: str | None = None
    """
    QOS description
    """
    flags: list[str] | None = None
    """
    List of properties of QOS
    """
    id: str | None = None
    """
    Database id
    """
    limits: Limits1 | None = None
    """
    Assigned limits
    """
    name: str | None = None
    """
    Assigned name of QOS
    """
    preempt: Preempt | None = None
    """
    Preemption settings
    """
    priority: int | None = None
    """
    QOS priority
    """
    usage_factor: float | None = None
    """
    Usage factor
    """
    usage_threshold: float | None = None
    """
    Usage threshold
    """


class DbQosInfo(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None
    qos: list[DbQos] | None = None
    """
    Array of QOS
    """


class DbResponseAccountDelete(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbResponseAssociations(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbResponseAssociationsDelete(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None
    removed_associations: list[str] | None = None
    """
    the associations
    """


class DbResponseClusterAdd(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbResponseClusterDelete(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbResponseQos(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbResponseQosDelete(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbResponseTres(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbResponseUserDelete(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbResponseUserUpdate(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbResponseWckeyAdd(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbResponseWckeyDelete(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None


class DbTresInfo(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None
    tres: DbTresList | None = None


class DbUpdateAccount(BaseModel):
    accounts: list[DbAccount] | None = None


class DbUpdateQos(BaseModel):
    qos: list[DbQos] | None = None


class DbUpdateUsers(BaseModel):
    users: list[DbUser] | None = None


class DbUserInfo(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None
    users: list[DbUser] | None = None
    """
    Array of users
    """


class DbWckey(BaseModel):
    accounting: list[DbAccounting] | None = None
    """
    List of accounting records
    """
    cluster: str | None = None
    """
    Cluster name
    """
    flags: list[str] | None = None
    """
    List of properties of wckey
    """
    id: int | None = None
    """
    wckey database unique id
    """
    name: str | None = None
    """
    wckey name
    """
    user: str | None = None
    """
    wckey user
    """


class DbWckeyInfo(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None
    wckeys: list[DbWckey] | None = None
    """
    List of wckeys
    """


class DbClusterInfo(BaseModel):
    associations: Associations | None = None
    """
    Information about associations
    """
    controller: Controller | None = None
    """
    Information about controller
    """
    flags: list[str] | None = None
    """
    List of properties of cluster
    """
    name: str | None = None
    """
    Cluster name
    """
    nodes: str | None = None
    """
    Assigned nodes
    """
    rpc_version: int | None = None
    """
    Number rpc version
    """
    select_plugin: str | None = None
    """
    Configured select plugin
    """
    tres: list[DbResponseTres] | None = None
    """
    List of TRES in cluster
    """


class DbClustersProperties(BaseModel):
    clusters: DbClusterInfo | None = None


class DbConfigInfo(BaseModel):
    accounts: list[DbAccount] | None = None
    """
    Array of accounts
    """
    associations: list[DbAssociation] | None = None
    """
    Array of associations
    """
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    meta: Meta | None = None
    qos: list[DbQos] | None = None
    """
    Array of qos
    """
    tres: list[DbTresList] | None = None
    """
    Array of TRES
    """
    users: list[DbUser] | None = None
    """
    Array of users
    """
    wckeys: list[DbWckey] | None = None
    """
    Array of wckeys
    """


class DbJob(BaseModel):
    """
    Single job description
    """

    account: str | None = None
    """
    Account charged by job
    """
    allocation_nodes: str | None = None
    """
    Nodes allocated to job
    """
    array: Array | None = None
    """
    Array properties (optional)
    """
    association: DbAssociationShortInfo | None = None
    cluster: str | None = None
    """
    Assigned cluster
    """
    comment: Comment | None = None
    """
    Job comments by type
    """
    constraints: str | None = None
    """
    Constraints on job
    """
    container: str | None = None
    """
    absolute path to OCI container bundle
    """
    derived_exit_code: DbJobExitCode | None = None
    exit_code: DbJobExitCode | None = None
    flags: list[str] | None = None
    """
    List of properties of job
    """
    group: str | None = None
    """
    User's group to run job
    """
    het: Het | None = None
    """
    Heterogeneous Job details (optional)
    """
    job_id: int | None = None
    """
    Job id
    """
    kill_request_user: str | None = None
    """
    User who requested job killed
    """
    mcs: Mcs | None = None
    """
    Multi-Category Security
    """
    name: str | None = None
    """
    Assigned job name
    """
    nodes: str | None = None
    """
    List of nodes allocated for job
    """
    partition: str | None = None
    """
    Assigned job's partition
    """
    priority: int | None = None
    """
    Priority
    """
    qos: str | None = None
    """
    Assigned qos name
    """
    required: Required | None = None
    """
    Job run requirements
    """
    reservation: Reservation | None = None
    """
    Reservation usage details
    """
    state: State | None = None
    """
    State properties of job
    """
    steps: list[DbJobStep] | None = None
    """
    Job step description
    """
    time: Time2 | None = None
    """
    Time properties
    """
    tres: Tres1 | None = None
    """
    TRES settings
    """
    user: str | None = None
    """
    Job user
    """
    wckey: Wckey | None = None
    """
    Job assigned wckey details
    """
    working_directory: str | None = None
    """
    Directory where job was initially started
    """


class DbJobInfo(BaseModel):
    errors: list[DbError] | None = None
    """
    Slurm errors
    """
    jobs: list[DbJob] | None = None
    """
    Array of jobs
    """
    meta: Meta | None = None


class DbSetConfig(BaseModel):
    TRES: list[DbTresList] | None = None
    accounts: list[DbUpdateAccount] | None = None
    associations: list[DbAssociation] | None = None
    clusters: list[DbClustersProperties] | None = None
    qos: list[DbQos] | None = None
    users: list[DbUser] | None = None
    wckeys: list[DbWckey] | None = None
