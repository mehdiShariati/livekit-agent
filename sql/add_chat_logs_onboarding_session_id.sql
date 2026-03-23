-- Optional: enables log lookup by onboarding session when room_name alone is insufficient.
-- Run against the same Postgres database used by LIVEKIT_AGENT / chat_logs.
ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS onboarding_session_id VARCHAR(64);

CREATE INDEX IF NOT EXISTS idx_chat_logs_onboarding_session_id
  ON chat_logs (onboarding_session_id)
  WHERE onboarding_session_id IS NOT NULL;
