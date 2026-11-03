# Foreground completion helper (candidate, not yet production accepted)

`hybrid_tsad.foreground_completion` observes existing caller-owned `Popen`
handles. It does not launch or authorize jobs. The caller must first verify the
root-approved argv, source/config/release identities, fresh output and applicable
resource limits. There is no automatic retry, scientific decision, timer service
or remote process termination.

Keep the actual foreground supervisor/SSH process: use `shell=False`, durable
stdout/stderr files (never undrained `PIPE`), record `time.monotonic()` at launch,
complete and close stdin, then register `ForegroundJob`. For SSH the approved
argv must explicitly include positive `ConnectTimeout`, `ServerAliveInterval`,
`ServerAliveCountMax` and `BatchMode=yes`. Authorization remains the caller's
responsibility; an arbitrary string passed as `approval_reference` is not proof
of approval. Preserve the original approval artifact and exact argv separately.

Each job specifies terminal metadata locations, its total transport wait limit,
a one-shot `read_terminal()` callback and the expected root identity fields
(normally `release_sha256` and `config_sha256`). The reader is called once after
the real process exits, including a process that exited before registration.
Use a bounded single metadata read of the saved `OS_WAIT.json`, not a periodic
OS-file poller. The helper itself performs no filesystem or SSH discovery.

Consume `observe_foreground(jobs, observer_seconds)` and durably write/send each
yielded event immediately. Independent wait threads prevent a slow job from
holding up a completed one. The observer uses an absolute monotonic wall
deadline and gives already-queued events priority. If the caller pauses between
yields or stops iterating, delivery is delayed; the generator cannot force
external delivery by that wall deadline. A blocked metadata
callback runs in a daemon thread and may outlive collection; the callback must
also bound its own I/O. An unresolved process is left intact for explicit root
review, with its PID, exact approved argv and terminal locations in the event.
Keeping a process handle or receiving empty stdout does not establish progress.

`COMPLETED` requires all of: transport exit 0, a saved terminal record with the
expected identities, scientific `OS_exit_code=0`, `resource_status=COMPLETED`,
and `cleanup.status=NO_LIVE_GROUP_MEMBERS` with `cleanup.live_remaining=[]`.
A contradictory record claiming no live members while listing live processes
is unresolved. A supervisor that itself exits 0
while its child failed is reported `FAILED`. Missing/invalid metadata, nonzero
transport, observation errors and timeouts are `UNRESOLVED`; none authorize a
remote restart. Events include actual observation UTC, not a backdated send time.

Validation is limited to thirteen local synthetic subprocess fixtures, including
three concurrent children, pre-registration exit, failure/disconnection/deadline
cases and a blocked reader. No real SSH/scientific job was used. This does not
yet demonstrate reliability under actual network failures. Old collector code,
logs and all scientific snapshots remain unchanged. Production use waits for
root and independent review of this candidate.
