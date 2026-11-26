from ggTrader.utils.db_snapshot import DIGEST_TABLES, snapshot, table_counts, table_digests


class Cursor:
    def __init__(self, log=None):
        self.log = log if log is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, *args):
        self.query = str(query)
        self.log.append((self.query, args))

    def fetchall(self):
        return [
            ("public", "items"),
            ("pg_catalog", "internal"),
            ("information_schema", "tables"),
            ("_timescaledb_internal", "x"),
        ]

    def fetchone(self):
        if "current_setting" in self.query:
            return ("America/Los_Angeles",)
        return (7,) if "count(*)" in self.query else ("digest",)


class Connection:
    def __init__(self):
        self.log = []

    def cursor(self):
        return Cursor(self.log)


def test_table_counts_skips_system_and_timescale_schemas():
    assert table_counts(Connection()) == {"public.items": 7}


def test_table_digests_covers_every_digest_table():
    assert table_digests(Connection()) == dict.fromkeys(DIGEST_TABLES, "digest")


def test_snapshot_has_counts_and_digests_keys():
    assert set(snapshot(Connection())) == {"counts", "digests"}


def test_digests_pin_utc_and_restore_callers_time_zone():
    conn = Connection()
    table_digests(conn)
    queries = [(q, a) for q, a in conn.log if "set_config" in q]
    assert queries[0][0].endswith("'UTC', false)")
    assert queries[-1][1] == (("America/Los_Angeles",),)
