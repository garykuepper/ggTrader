# Database dump timer

The owner installed these user-level units on 2026-11-26. The timer runs Sunday at 04:30 local
time under the shared capped-job guard and keeps four dumps. Check its schedule with:

```sh
systemctl --user list-timers ggtrader-db-dump.timer
```

The templates use `__REPO_ROOT__` as the checkout path and require the output folder on thesix
to exist. `SHA256SUMS` retains lines for pruned dumps; this is expected.
