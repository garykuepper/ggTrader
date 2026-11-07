#!/usr/bin/env bash
# Export ONLY the DB_* / POSTGRES_* keys from the production .env (read-only) into the
# environment of the wrapped command. Values are never printed. Alpaca/Telegram keys
# are deliberately NOT exported. Usage: with_db_env.sh <cmd> [args...]
set -euo pipefail
ENV_FILE=/home/flynn/ggTrader/.env
while IFS='=' read -r k v; do
    case "$k" in
        DB_USER|DB_PASS|DB_HOST|DB_NAME|DB_PORT|POSTGRES_CONNECTION_STRING)
            v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
            export "$k=$v" ;;
    esac
done < <(grep -E '^(DB_USER|DB_PASS|DB_HOST|DB_NAME|DB_PORT|POSTGRES_CONNECTION_STRING)=' "$ENV_FILE")
# host-side runs use localhost:5433 per CLAUDE.md
if [[ "${DB_HOST:-}" == "host.docker.internal" ]]; then export DB_HOST=localhost; export DB_PORT=5433; fi
exec "$@"
