-- Bootstraps a read-only Postgres role used by execute_scoped_sql.
--
-- LOCAL DEV: mounted at /docker-entrypoint-initdb.d/ in docker-compose so it
--   runs once when the volume is first initialized.
-- SUPABASE:  run this in the Supabase SQL editor after init_db() has created
--   the tables. Idempotent — safe to re-run.
--
-- The role has SELECT on the five rent-roll tables and nothing else.
-- Defense-in-depth: even if sqlglot is bypassed, the DB rejects writes.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'property_reader') THEN
    CREATE ROLE property_reader LOGIN PASSWORD 'reader_pass';
  END IF;
END
$$;

-- Tables may not exist yet (first run, before init_db()). Use a DO block that
-- silently skips missing tables so this script can be re-run any time.
DO $$
DECLARE
  t TEXT;
  rent_tables TEXT[] := ARRAY[
    'properties', 'units', 'leases', 'rent_snapshots', 'rent_charge_lines'
  ];
BEGIN
  GRANT USAGE ON SCHEMA public TO property_reader;
  FOREACH t IN ARRAY rent_tables LOOP
    IF EXISTS (
      SELECT 1 FROM information_schema.tables
      WHERE table_schema = 'public' AND table_name = t
    ) THEN
      EXECUTE format('GRANT SELECT ON public.%I TO property_reader', t);
    END IF;
  END LOOP;
END
$$;

-- Future tables created by init_db() also need the grant. Set default privs
-- so any table created later by the `property_user` role automatically
-- exposes SELECT to property_reader.
ALTER DEFAULT PRIVILEGES FOR ROLE property_user IN SCHEMA public
  GRANT SELECT ON TABLES TO property_reader;
