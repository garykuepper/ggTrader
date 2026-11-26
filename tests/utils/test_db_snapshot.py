from ggTrader.utils.db_snapshot import DIGEST_TABLES, snapshot, table_counts, table_digests


class Cursor:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, *args):
        self.query = str(query)

    def fetchall(self):
        return [
            ("public", "items"),
            ("pg_catalog", "internal"),
            ("information_schema", "tables"),
            ("_timescaledb_internal", "x"),
        ]

    def fetchone(self):
        return (7,) if "count(*)" in self.query else ("digest",)


class Connection:
    def cursor(self):
        return Cursor()


def test_table_counts_skips_system_and_timescale_schemas():
    assert table_counts(Connection()) == {"public.items": 7}


def test_table_digests_covers_every_digest_table():
    assert table_digests(Connection()) == dict.fromkeys(DIGEST_TABLES, "digest")


def test_snapshot_has_counts_and_digests_keys():
    assert set(snapshot(Connection())) == {"counts", "digests"}
