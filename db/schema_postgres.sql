-- Dalilak AI — PostgreSQL Schema
-- STATUS: CODE-ONLY — DO NOT EXECUTE ON PRODUCTION until:
--   1. Qdrant backup snapshot taken and verified
--   2. Dry-run on staging environment
--   3. Owner-approved rollback plan in place
-- ─────────────────────────────────────────────────────────────────────────────

-- EXTENSIONS
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ─────────────────────────────────────────────────────────────────────────────
-- USERS
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      VARCHAR(64)  NOT NULL UNIQUE,
    email         VARCHAR(255) NOT NULL UNIQUE,
    password_hash TEXT         NOT NULL,
    is_admin      BOOLEAN      NOT NULL DEFAULT FALSE,
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    plan          VARCHAR(32)  NOT NULL DEFAULT 'free',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email    ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

-- ─────────────────────────────────────────────────────────────────────────────
-- PLANS & QUOTAS
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS plan_configs (
    plan            VARCHAR(32) PRIMARY KEY,
    daily_messages  INTEGER NOT NULL DEFAULT 10,
    monthly_messages INTEGER NOT NULL DEFAULT 300,
    max_tokens      INTEGER NOT NULL DEFAULT 1024,
    features        JSONB   NOT NULL DEFAULT '{}'
);

INSERT INTO plan_configs (plan, daily_messages, monthly_messages, max_tokens)
VALUES
    ('free',  10,  300,  1024),
    ('pro',   100, 3000, 4096),
    ('admin', 9999, 999999, 8192)
ON CONFLICT (plan) DO NOTHING;

CREATE TABLE IF NOT EXISTS user_quotas (
    user_id       UUID        REFERENCES users(id) ON DELETE CASCADE,
    period_start  DATE        NOT NULL,
    period_type   VARCHAR(8)  NOT NULL CHECK (period_type IN ('daily','monthly')),
    usage_count   INTEGER     NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, period_start, period_type)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- PASSWORD RESETS
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS password_resets (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT        NOT NULL UNIQUE,
    expires_at  TIMESTAMPTZ NOT NULL,
    used_at     TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_resets_token_hash ON password_resets (token_hash);
CREATE INDEX IF NOT EXISTS idx_resets_user_id    ON password_resets (user_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- JWT JTI REVOCATION (durable, Redis is the operational store)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS revoked_jtis (
    jti         TEXT        PRIMARY KEY,
    user_id     UUID        REFERENCES users(id) ON DELETE SET NULL,
    revoked_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jtis_expires ON revoked_jtis (expires_at);

-- ─────────────────────────────────────────────────────────────────────────────
-- USER PROCEDURES (My Procedure workspace)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_procedures (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    procedure_id   VARCHAR(128) NOT NULL,
    title_ar       TEXT        NOT NULL,
    status         VARCHAR(32) NOT NULL DEFAULT 'active'
                               CHECK (status IN ('active','completed','cancelled')),
    notes          TEXT        NOT NULL DEFAULT '',
    completed_steps JSONB      NOT NULL DEFAULT '[]',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_uproc_user ON user_procedures (user_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- CONTENT GOVERNANCE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TYPE content_status AS ENUM ('draft','review','approved','published','expired');

CREATE TABLE IF NOT EXISTS content_items (
    id              UUID           PRIMARY KEY DEFAULT gen_random_uuid(),
    title_ar        TEXT           NOT NULL,
    body_ar         TEXT           NOT NULL,
    content_type    VARCHAR(64)    NOT NULL DEFAULT 'procedure_update',
    ref_procedure_id VARCHAR(128),
    status          content_status NOT NULL DEFAULT 'draft',
    created_by      UUID           REFERENCES users(id) ON DELETE SET NULL,
    published_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- AUDIT LOG
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id         BIGSERIAL   PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action     VARCHAR(64) NOT NULL,
    item_id    TEXT        NOT NULL,
    actor_id   UUID        REFERENCES users(id) ON DELETE SET NULL,
    before     TEXT,
    after      TEXT,
    note       TEXT        NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_audit_item  ON audit_log (item_id);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log (actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts    ON audit_log (ts DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- HELPER: auto-update updated_at
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_uproc_updated_at
    BEFORE UPDATE ON user_procedures
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_content_updated_at
    BEFORE UPDATE ON content_items
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
