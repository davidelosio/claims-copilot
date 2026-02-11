/* Claims Copilot — PostgreSQL schema */
/* Run: psql claims_copilot < sql/schema.sql */

BEGIN;

/* ============================================================ */
/* CORE TABLES */
/* ============================================================ */

CREATE TABLE IF NOT EXISTS policyholders (
    policyholder_id     TEXT PRIMARY KEY,
    first_name          TEXT NOT NULL,
    last_name           TEXT NOT NULL,
    date_of_birth       DATE NOT NULL,
    gender              TEXT, /* M / F / Other */
    city                TEXT NOT NULL,
    province            TEXT NOT NULL, /* Italian province code */
    driving_license_year INT NOT NULL, /* year license was obtained */
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id          TEXT PRIMARY KEY,
    policyholder_id     TEXT NOT NULL REFERENCES policyholders(policyholder_id),
    make                TEXT NOT NULL,
    model               TEXT NOT NULL,
    year                INT NOT NULL,
    fuel_type           TEXT, /* petrol / diesel / electric / hybrid */
    estimated_value     NUMERIC(10,2) NOT NULL,
    plate_number        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id           TEXT PRIMARY KEY,
    policyholder_id     TEXT NOT NULL REFERENCES policyholders(policyholder_id),
    vehicle_id          TEXT NOT NULL REFERENCES vehicles(vehicle_id),
    policy_type         TEXT NOT NULL, /* third_party / comprehensive / fire_theft */
    inception_date      DATE NOT NULL,
    expiry_date         DATE NOT NULL,
    annual_premium      NUMERIC(10,2) NOT NULL,
    coverage_limit      NUMERIC(12,2) NOT NULL,
    deductible          NUMERIC(10,2) DEFAULT 0
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id            TEXT PRIMARY KEY,
    policy_id           TEXT NOT NULL REFERENCES policies(policy_id),
/* incident info */
    incident_date       DATE NOT NULL,
    incident_time       TIME,
    incident_city       TEXT,
    incident_province   TEXT,
    incident_type       TEXT NOT NULL, /* collision / theft / vandalism / weather / single_vehicle / hit_and_run / parking */
    description         TEXT NOT NULL, /* free-text from policyholder */
/* parties & damage */
    num_parties         INT DEFAULT 1,
    injuries            BOOLEAN DEFAULT FALSE,
    injury_severity     TEXT, /* none / minor / moderate / severe */
    police_report       BOOLEAN DEFAULT FALSE,
    damage_estimate     NUMERIC(10,2),
/* status tracking */
    status              TEXT DEFAULT 'new', /* new / in_progress / pending_docs / settled / denied / reopened */
    created_at          TIMESTAMPTZ DEFAULT now(),
    first_contact_at    TIMESTAMPTZ,
    settled_at          TIMESTAMPTZ,
/* handler assignment */
    assigned_handler    TEXT,
    handler_queue       TEXT /* fast_lane / standard / specialist / fraud_review */
);

/* Documents / attachments metadata */
CREATE TABLE IF NOT EXISTS claim_documents (
    document_id         TEXT PRIMARY KEY,
    claim_id            TEXT NOT NULL REFERENCES claims(claim_id),
    doc_type            TEXT NOT NULL, /* photo / police_report / repair_estimate / medical_report / id_document / other */
    uploaded_at         TIMESTAMPTZ DEFAULT now(),
    present             BOOLEAN DEFAULT TRUE /* whether it's actually uploaded or still missing */
);

/* ============================================================ */
/* GROUND TRUTH (only used for training, never shown to handler) */
/* ============================================================ */

CREATE TABLE IF NOT EXISTS claim_labels (
    claim_id            TEXT PRIMARY KEY REFERENCES claims(claim_id),
/* complexity */
    complexity          TEXT NOT NULL, /* simple / medium / complex */
    handling_days       INT NOT NULL, /* actual days to settle */
    num_touches         INT NOT NULL, /* number of handler actions needed */
/* fraud */
    is_fraud            BOOLEAN DEFAULT FALSE,
    fraud_type          TEXT, /* null / staged / inflated / phantom / repeated */
/* outcome */
    settled_amount      NUMERIC(10,2),
    was_reopened        BOOLEAN DEFAULT FALSE
);

/* ============================================================ */
/* EVENT LOG (for ML feature engineering with time correctness) */
/* ============================================================ */

CREATE TABLE IF NOT EXISTS claim_events (
    event_id            BIGSERIAL PRIMARY KEY,
    claim_id            TEXT NOT NULL REFERENCES claims(claim_id),
    event_type          TEXT NOT NULL, /* created / assigned / doc_uploaded / doc_requested / contacted_customer / note_added / escalated / settled / denied / reopened */
    event_timestamp     TIMESTAMPTZ NOT NULL,
    event_data          JSONB, /* flexible payload */
    actor               TEXT /* handler_id or 'system' */
);

/* ============================================================ */
/* COPILOT OUTPUTS & FEEDBACK (Layer 6) */
/* ============================================================ */

CREATE TABLE IF NOT EXISTS copilot_outputs (
    output_id           BIGSERIAL PRIMARY KEY,
    claim_id            TEXT NOT NULL REFERENCES claims(claim_id),
    created_at          TIMESTAMPTZ DEFAULT now(),
    summary             TEXT,
    extracted_facts     JSONB,
    complexity_score    NUMERIC(4,3),
    complexity_label    TEXT,
    fraud_score         NUMERIC(4,3),
    fraud_label         TEXT,
    next_actions        JSONB,
    model_versions      JSONB /* which model versions produced this */
);

CREATE TABLE IF NOT EXISTS copilot_feedback (
    feedback_id         BIGSERIAL PRIMARY KEY,
    output_id           BIGINT NOT NULL REFERENCES copilot_outputs(output_id),
    claim_id            TEXT NOT NULL REFERENCES claims(claim_id),
    handler_id          TEXT,
    feedback_type       TEXT NOT NULL, /* accepted / edited / rejected */
    feedback_detail     JSONB, /* what was changed */
    created_at          TIMESTAMPTZ DEFAULT now()
);

/* ============================================================ */
/* INDEXES */
/* ============================================================ */

CREATE INDEX IF NOT EXISTS idx_claims_policy ON claims(policy_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_incident_date ON claims(incident_date);
CREATE INDEX IF NOT EXISTS idx_claim_events_claim ON claim_events(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_events_ts ON claim_events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_vehicles_policyholder ON vehicles(policyholder_id);
CREATE INDEX IF NOT EXISTS idx_policies_policyholder ON policies(policyholder_id);

COMMIT;
