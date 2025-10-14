# Quick Start Guide - 5 Minutes to Production

Get your SLP SimuCase Generator running with Auth0 authentication in 5 minutes.

## Prerequisites

- Docker and Docker Compose installed
- An Auth0 account (free tier works fine)
- API keys for at least one LLM provider

---

## Step 1: Auth0 Setup (2 minutes)

### Create Application

1. Go to [Auth0 Dashboard](https://manage.auth0.com/)
2. Click **Applications > Create Application**
3. Name: `SLP SimuCase`
4. Type: **Regular Web Application**
5. Click **Create**

### Configure URLs

In your new application settings:
```
Allowed Callback URLs:
  http://localhost:4180/oauth2/callback

Allowed Logout URLs:
  http://localhost:4180/

Allowed Web Origins:
  http://localhost:4180
```

### Create API

1. Go to **Applications > APIs > Create API**
2. Name: `SLP SimuCase API`
3. Identifier: `https://api.slp-simucase.local`
4. Signing Algorithm: `RS256`

### Enable RBAC

1. In your API settings:
   - ✓ Enable RBAC
   - ✓ Add Permissions in the Access Token

2. Add these permissions:
   ```
   create:cases
   read:cases
   read:analytics
   ```

### Create a Role

1. Go to **User Management > Roles > Create Role**
2. Name: `Clinician`
3. Description: `Can generate cases`
4. Permissions: Select all 3 permissions

### Assign Role to Your User

1. Go to **User Management > Users**
2. Select your user
3. Click **Roles** tab
4. Click **Assign Roles**
5. Select `Clinician`

**Note your credentials:**
- Domain: `your-tenant.auth0.com`
- Client ID: `abc123...`
- Client Secret: `xyz789...`

---

## Step 2: Clone and Configure (1 minute)

```bash
# Clone repository
git clone https://github.com/yourusername/slp-simucase.git
cd slp-simucase

# Copy environment template
cp .env.example .env
```

Edit `.env`:
```bash
# Auth0 (from Step 1)
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
AUTH0_API_AUDIENCE=https://api.slp-simucase.local
AUTH0_ISSUER_URL=https://your-tenant.auth0.com/
OAUTH2_REDIRECT_URL=http://localhost:4180/oauth2/callback

# Generate cookie secret
OAUTH2_PROXY_COOKIE_SECRET=$(python -c 'import os,base64; print(base64.b64encode(os.urandom(32)).decode())')

# Add at least one API key
OPENAI_API_KEY=sk-your-key-here

# Use defaults for local development
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/slp_rbac
POSTGRES_PASSWORD=postgres
REDIS_URL=redis://:redis_password@redis:6379/0
REDIS_PASSWORD=redis_password
```

---

## Step 3: Launch (2 minutes)

```bash
# Start all services
docker-compose up -d

# Wait for services to be ready (30-60 seconds)
docker-compose ps

# Check logs
docker-compose logs -f
```

Expected output:
```
oauth2-proxy     | listening on http://0.0.0.0:4180
fastapi-backend  | Uvicorn running on http://0.0.0.0:8000
gradio-app       | Running on local URL:  http://0.0.0.0:7860
postgres         | database system is ready to accept connections
redis            | Ready to accept connections
```

---

## Step 4: Access and Test

1. **Open your browser**: http://localhost:4180
2. **Click "Sign in with Auth0"**
3. **Log in with your Auth0 credentials**
4. **You're in!** Start generating cases

### Test Case Generation

1. Click **"Generate Single Case"**
2. Select:
   - Grade: `1st Grade`
   - Disorders: `Articulation Disorders`
   - Model: `Llama3.2` (or GPT-4o if you have OpenAI key)
3. Click **"Generate"**
4. Wait 10-30 seconds
5. View your generated case!

---

## Troubleshooting

### "Invalid credentials" error
**Problem**: OAuth2 login fails
**Solution**:
1. Check Auth0 callback URLs match exactly: `http://localhost:4180/oauth2/callback`
2. Verify Client ID and Secret in `.env`
3. Check Auth0 application is enabled

### "Permission denied" error
**Problem**: Can't generate cases
**Solution**:
1. Verify role is assigned to your user in Auth0
2. Check role has `create:cases` permission
3. Try logging out and back in

### Container won't start
**Problem**: Docker service fails
**Solution**:
```bash
# Check logs
docker-compose logs <service-name>

# Common fixes:
# 1. Port already in use
docker-compose down
docker-compose up -d

# 2. Invalid environment variable
cat .env | grep -v '^#' | grep -v '^$'
```

### Can't connect to database
**Problem**: Database connection error
**Solution**:
```bash
# Wait for database to initialize
docker-compose logs postgres | grep "ready"

# Restart if needed
docker-compose restart fastapi-backend gradio-app
```

---

## What's Next?

### For Development
- Read the [Architecture Overview](architecture.md)
- Explore the [API Documentation](http://localhost:8000/api/docs)
- Set up additional LLM providers

### For Production
- Follow the [Deployment Guide](DEPLOYMENT.md)
- Configure SSL/TLS certificates
- Set up monitoring and backups
- Review [Security Best Practices](security.md)

### For Researchers
- Configure [Data Retention](compliance.md)
- Set up [Analytics Dashboard](analytics.md)
- Review [IRB Documentation](irb-template.md)

---

## Quick Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart a service
docker-compose restart <service-name>

# Update to latest version
git pull
docker-compose pull
docker-compose up -d

# Backup database
docker-compose exec postgres pg_dump -U postgres slp_rbac > backup.sql

# Clean up (removes all data!)
docker-compose down -v
```

---

## Support

- **Documentation**: See `docs/` folder
- **Issues**: https://github.com/yourusername/slp-simucase/issues
- **Auth0 Docs**: https://auth0.com/docs

---

**You're all set! Start generating cases! 🎉**
