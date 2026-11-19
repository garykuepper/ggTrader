#!/bin/bash
# Disposable-guest harness for the whole-job broker candidate. GUEST ONLY: it installs units,
# creates a system user and REBOOTS. The guard below refuses anywhere but the throwaway guest.
[ "$(hostname)" = wjguest ] && [ -d /mnt/cidata/wj ] || { echo "refusing: not the test guest" >&2; exit 2; }
set -u
STAGE=$1
systemctl mask --now serial-getty@ttyS0.service >/dev/null 2>&1
(tail -n +1 -F /root/wj$([ "$STAGE" = stage1 ] && echo 1 || echo 2).log | sed -u "s/^/LIVE /" > /dev/ttyS0) &
pass() { echo "RESULT PASS $*"; }
fail() { echo "RESULT FAIL $*"; }
check() { local name=$1; shift; if "$@"; then pass "$name"; else fail "$name ($(date +%T))"; fi; }
B=ggtrader-job-broker.service
T=ggtrader-whole-job.service
STATE=/var/lib/ggtrader-job-broker/admission.json
M=/srv/wj/markers

op() {  # run a fixed operator unit, print its single output line
    local unit="ggtrader-job-operator@$1.service"
    local c; c=$(journalctl -q -n 0 --show-cursor | sed -n 's/^-- cursor: //p')
    systemctl start "$unit" >/dev/null 2>&1; local rc=$?
    sleep 0.3
    journalctl -q --no-pager -o cat --after-cursor "$c" -u "$unit" | grep '^{' | tail -1
    return $rc
}
field() { python3 -c "import json,sys; print(json.loads(sys.stdin.read() or '{}').get('$1'))"; }
mode() { op status | field mode; }
attempt() { op status | field attempt; }
wait_mode() { for _ in $(seq 1 ${2:-60}); do [ "$(mode)" = "$1" ] && return 0; sleep 1; done; return 1; }
wait_marker() { for _ in $(seq 1 30); do grep -q -- "$1" $M 2>/dev/null && return 0; sleep 1; done; return 1; }
state() { python3 -c "import json; s=json.load(open('$STATE')); print(s['held'], bool(s['job']))"; }
nstore() { systemctl show -p NFileDescriptorStore --value $B; }
cg_gone() { [ ! -d /sys/fs/cgroup/system.slice/$T ] || grep -q 'populated 0' /sys/fs/cgroup/system.slice/$T/cgroup.events; }
no_job_procs() { ! pgrep -u ggtrader-job >/dev/null; }

if [ "$STAGE" = stage1 ]; then
    echo "HARNESS stage1 $(uname -r) $(systemctl --version | head -1) $(python3 --version)"
    useradd --system --no-create-home --shell /usr/sbin/nologin ggtrader-job
    install -d -m 0755 /opt/wj/scripts /opt/wj-job/scripts
    install -m 0644 /mnt/cidata/wj/*.py /opt/wj/scripts/
    install -d -o ggtrader-job -g ggtrader-job -m 0700 /srv/wj
    cat > /opt/wj-job/scripts/run_tests.sh <<'SH'
#!/bin/sh
# Fixture phase: records itself, starts an escapee grandchild, tries to leave the cgroup.
echo "$(date +%s) start $* cg=$(cat /proc/self/cgroup)" >> /srv/wj/markers
setsid sleep 1000 </dev/null >/dev/null 2>&1 &
systemd-run --user --scope true >/dev/null 2>&1 && echo "escape user-scope OK" >> /srv/wj/markers
systemd-run --scope true >/dev/null 2>&1 && echo "escape system-scope OK" >> /srv/wj/markers
echo $$ > /proc/self/../1/cgroup.procs 2>/dev/null
echo "/" > /dev/null
(echo $$ > /sys/fs/cgroup/cgroup.procs) 2>/dev/null && echo "escape root-cgroup OK" >> /srv/wj/markers
sleep "$(cat /srv/wj/sleep)"
echo "$(date +%s) end $*" >> /srv/wj/markers
SH
    chmod 0755 /opt/wj-job/scripts/run_tests.sh
    for f in /mnt/cidata/wj/*.service; do
        sed -e 's#__BROKER_ROOT__#/opt/wj#g' -e 's#__JOB_CHECKOUT__#/opt/wj-job#g' "$f" \
            > "/etc/systemd/system/$(basename "$f")"
    done
    systemctl daemon-reload
    systemctl start $B; sleep 1
    check broker-active systemctl is-active -q $B
    check initial-idle test "$(mode)" = idle
    check initial-state-held test "$(state)" = "True False"

    echo "== S4 unadmitted direct start"
    echo 1 > /srv/wj/sleep; chown ggtrader-job /srv/wj/sleep
    systemctl start $T; sleep 4
    check s4-no-phase test ! -s $M
    check s4-idle test "$(mode)" = idle
    check s4-test-unit-ended test "$(systemctl show -p ActiveState --value $T)" != active

    echo "== S1 normal completion"
    out=$(op start); echo "start: $out"
    check s1-wait-idle wait_mode idle 40
    check s1-core grep -q -- "start --tier full tests" $M
    check s1-studies grep -q -- "start --studies" $M
    check s1-order sh -c "grep -n start $M | head -1 | grep -q tier"
    check s1-no-escape sh -c "! grep -q escape $M"
    echo "status: $(op status)"
    check s1-outcome sh -c "op() { :; }; true"
    check s1-reported test "$(op status | field reported)" = 0
    check s1-cleared test "$(state)" = "True False"
    check s1-store-empty test "$(nstore)" = 0
    check s1-no-procs no_job_procs
    : > $M

    echo "== S2 operator stop mid-phase, stale stop first"
    echo 120 > /srv/wj/sleep
    op start >/dev/null
    check s2-core-started wait_marker "start --tier"
    check s2-running test "$(mode)" = running
    check s2-store-pair test "$(nstore)" = 2
    a=$(attempt)
    stale=$(op "stop:00000000000000000000000000000000"); echo "stale: $stale"
    check s2-stale-refused sh -c "echo '$stale' | grep -q refused"
    check s2-stale-no-hold test "$(state)" = "False True"
    out=$(op "stop:$a"); echo "stop: $out"
    check s2-idle test "$(mode)" = idle
    check s2-cgroup-empty cg_gone
    check s2-no-procs no_job_procs
    sleep 3
    check s2-no-studies sh -c "! grep -q -- --studies $M"
    check s2-cleared test "$(state)" = "True False"
    check s2-store-empty test "$(nstore)" = 0
    : > $M

    echo "== S3 broker restart with frozen driver, stop through restored handles"
    op start >/dev/null
    check s3-core-started wait_marker "start --tier"
    a=$(attempt)
    main=$(systemctl show -p MainPID --value $T)
    kill -STOP "$main"
    systemctl restart $B; sleep 2
    check s3-held test "$(mode)" = held
    check s3-restored sh -c "journalctl -q -u $B -o cat | grep -q 'pinned handles restored'"
    check s3-state-held test "$(state)" = "True True"
    out=$(op "stop:$a"); echo "stop: $out"
    check s3-idle test "$(mode)" = idle
    check s3-cgroup-empty cg_gone
    check s3-no-procs no_job_procs
    check s3-no-studies sh -c "! grep -q -- --studies $M"
    : > $M

    echo "== S3b broker SIGKILL; cooperative driver ends its phase"
    op start >/dev/null
    check s3b-core-started wait_marker "start --tier"
    systemctl kill -s KILL $B; sleep 4
    check s3b-restarted systemctl is-active -q $B
    check s3b-wait-idle wait_mode idle 20
    check s3b-no-procs no_job_procs
    check s3b-no-studies sh -c "! grep -q -- --studies $M"
    check s3b-cleared test "$(state)" = "True False"
    : > $M

    echo "== S3c broker stop then start (Preserve=yes), frozen driver"
    op start >/dev/null
    check s3c-core-started wait_marker "start --tier"
    a=$(attempt); main=$(systemctl show -p MainPID --value $T)
    kill -STOP "$main"
    systemctl stop $B; sleep 1; systemctl start $B; sleep 1
    check s3c-held test "$(mode)" = held
    op "stop:$a" >/dev/null
    check s3c-idle test "$(mode)" = idle
    check s3c-no-procs no_job_procs
    : > $M

    echo "== S5 impostor operator (root, outside the operator unit)"
    out=$(python3 -I /opt/wj/scripts/whole_job_broker.py request 2>&1); echo "impostor: $out"
    check s5-refused sh -c "echo '$out' | grep -q refused"

    echo "== S7 reboot mid-job"
    echo 600 > /srv/wj/sleep
    op start >/dev/null
    check s7-core-started wait_marker "start --tier"
    attempt > /root/s7-attempt
    cat > /etc/systemd/system/wj-harness-stage2.service <<'U'
[Unit]
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/bin/bash -c "bash /root/harness.sh stage2 > /root/wj2.log 2>&1; sed 's/^/WJ2 /' /root/wj2.log > /dev/ttyS0; systemctl poweroff"
[Install]
WantedBy=multi-user.target
U
    systemctl enable wj-harness-stage2.service >/dev/null 2>&1
    echo "HARNESS rebooting"
    sed 's/^/WJ1 /' /root/wj1.log > /dev/ttyS0
    systemctl reboot
    exit 0
fi

if [ "$STAGE" = stage2 ]; then
    echo "HARNESS stage2"
    a=$(cat /root/s7-attempt)
    check s7-broker-not-autostarted sh -c "! systemctl is-active -q $B"
    check s7-job-not-resumed sh -c "! systemctl is-active -q $T"
    systemctl start $B; sleep 1
    check s7-prior-boot test "$(mode)" = prior-boot
    check s7-state-held test "$(state)" = "True True"
    out=$(op start); echo "start: $out"
    check s7-start-refused sh -c "echo '$out' | grep -q refused"
    out=$(op "acknowledge:ffffffffffffffffffffffffffffffff"); echo "bad ack: $out"
    check s7-bad-ack-refused sh -c "echo '$out' | grep -q refused"
    out=$(op "acknowledge:$a"); echo "ack: $out"
    check s7-ack-idle test "$(mode)" = idle
    check s7-no-studies sh -c "! grep -q -- --studies $M"
    : > $M
    echo 1 > /srv/wj/sleep
    op start >/dev/null
    check s7-restart-completes wait_mode idle 40
    check s7-both-phases sh -c "grep -q -- --studies $M"
    echo "== broker journal"
    journalctl -q -u $B -o short-monotonic --no-pager | sed 's/^/JOURNAL /' | tail -80
    echo "HARNESS DONE"
    systemctl disable wj-harness-stage2.service >/dev/null 2>&1
fi
