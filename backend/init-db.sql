-- Initial database setup for SLP SimuCase RBAC
-- This script runs automatically when PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search optimization

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE slp_rbac TO postgres;

-- Audit logging function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create indexes for performance
-- These will be applied to tables created by SQLAlchemy
