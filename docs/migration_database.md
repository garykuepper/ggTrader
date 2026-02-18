# Export TimescaleDB to Linux Server

This guide outlines the steps to export the `ggtrader` database from the local Docker container on Windows and migrate it to a remote Linux server.

## Migration Steps

### 1. Export Database locally
The most reliable way to export while seeing progress is via the provided Python utility. This script handles binary output safely and streams progress to your terminal.

```powershell
python scripts/db/export_db.py
```

- **What it does**: Runs `pg_dump -v` inside the `ggtrader_db` container and saves it to `ggtrader_dump.bak`.
- **Progress**: It will print every table and object being dumped in real-time.
- **Manual Command (Alternative)**:
  If you prefer raw commands, use:
  `cmd /c "docker exec -t ggtrader_db pg_dump -U ggtrader -v -Fc ggtrader > ggtrader_dump.bak"`

### 2. Transfer to Linux Server
Use `scp` to move the backup file to your remote server.

```powershell
scp ggtrader_dump.bak <user>@<server-ip>:/home/<user>/
```

### 3. Import on Linux Server
Log into your Linux server and use `pg_restore` to import the data.

#### Case A: Remote Database is in Docker
```bash
cat ggtrader_dump.bak | docker exec -i <remote_container_name> pg_restore -U ggtrader -d ggtrader --clean --if-exists
```

#### Case B: Remote Database is Native
```bash
pg_restore -U ggtrader -d ggtrader --clean --if-exists ggtrader_dump.bak
```

> [!IMPORTANT]
> Ensure the target database `ggtrader` exists and the TimescaleDB extension is installed on the remote server before restoring.

## Verification
Run a simple query on the remote server after migration to verify data presence:

```sql
SELECT count(*) FROM symbols;
```
