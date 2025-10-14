# Production Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Auth0 Configuration](#auth0-configuration)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [SSL/TLS Configuration](#ssltls-configuration)
6. [Database Setup](#database-setup)
7. [Monitoring Setup](#monitoring-setup)
8. [Backup Strategy](#backup-strategy)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- Docker 20.10+
- Docker Compose 2.0+
- Domain name with DNS configured
- Auth0 account
- API keys for LLM providers

### For Kubernetes
- Kubernetes 1.24+
- kubectl configured
- Helm 3+ (optional)
- Ingress controller (nginx)
- cert-manager for SSL

---

## Auth0 Configuration

### Step 1: Create Auth0 Tenant

1. Go to https://auth0.com and create an account
2. Create a new tenant (e.g., `your-org-research`)
3. Note your tenant domain: `your-org-research.auth0.com`

### Step 2: Create Application

1. **Applications > Create Application**
2. Name: `SLP SimuCase Generator`
3. Type: `Regular Web Application`
4. Technology: `Generic`

5. Configure Application Settings:
```
Name: SLP SimuCase Generator
Application Type: Regular Web Application

# URLs
Application Login URI: https://your-domain.com
Allowed Callback URLs:
  https://your-domain.com/oauth2/callback
  http://localhost:4180/oauth2/callback

Allowed Logout URLs:
  https://your-domain.com
  http://localhost:4180

Allowed Web Origins:
  https://your-domain.com
  http://localhost:4180

# Advanced Settings
Grant Types:
  ✓ Authorization Code
  ✓ Refresh Token

# Token Settings
Rotation: ✓ Enabled
Absolute Lifetime: 2592000 seconds (30 days)
```

6. Save and note:
   - **Client ID**
   - **Client Secret**
   - **Domain**

### Step 3: Create API

1. **Applications > APIs > Create API**
2. Name: `SLP SimuCase API`
3. Identifier: `https://api.slp-simucase.com` (can be any URL format)
4. Signing Algorithm: `RS256`

5. Enable RBAC:
```
Settings:
  ✓ Enable RBAC
  ✓ Add Permissions in the Access Token
```

6. Add Permissions (Scopes):
```
Permission              Description
-----------------------------------------------------------------
create:cases            Generate new case files
read:cases              View case files
update:cases            Edit case files
delete:cases            Delete case files
read:analytics          Access analytics dashboard
export:data             Export case data
manage:users            Manage user accounts
view:audit_logs         View audit logs
```

### Step 4: Create Roles

1. **User Management > Roles > Create Role**

Create these roles:

**Admin Role**
```
Name: Admin
Description: Full system access
Permissions:
  ✓ create:cases
  ✓ read:cases
  ✓ update:cases
  ✓ delete:cases
  ✓ read:analytics
  ✓ export:data
  ✓ manage:users
  ✓ view:audit_logs
```

**Researcher Role**
```
Name: Researcher
Description: Research and analytics access
Permissions:
  ✓ create:cases
  ✓ read:cases
  ✓ read:analytics
  ✓ export:data
```

**Clinician Role**
```
Name: Clinician
Description: Case generation access
Permissions:
  ✓ create:cases
  ✓ read:cases
```

**Viewer Role**
```
Name: Viewer
Description: Read-only access
Permissions:
  ✓ read:cases
```

### Step 5: Assign Roles to Users

1. **User Management > Users**
2. Select a user
3. Click **Roles** tab
4. Click **Assign Roles**
5. Select appropriate role(s)

---

## Docker Deployment

### Step 1: Clone and Configure

```bash
git clone https://github.com/yourusername/slp-simucase.git
cd slp-simucase

# Copy environment template
cp .env.example .env
```

### Step 2: Configure Environment

Edit `.env` with your values:

```bash
# Auth0 Configuration
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=<from Auth0 application>
AUTH0_CLIENT_SECRET=<from Auth0 application>
AUTH0_API_AUDIENCE=https://api.slp-simucase.com
AUTH0_ISSUER_URL=https://your-tenant.auth0.com/
OAUTH2_REDIRECT_URL=https://your-domain.com/oauth2/callback

# Generate cookie secret
OAUTH2_PROXY_COOKIE_SECRET=$(python -c 'import os,base64; print(base64.b64encode(os.urandom(32)).decode())')

# Database
POSTGRES_PASSWORD=<generate strong password>

# Redis
REDIS_PASSWORD=<generate strong password>

# API Keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...

# Security
ALLOWED_ORIGINS=https://your-domain.com
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
```

### Step 3: Launch Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f oauth2-proxy
docker-compose logs -f fastapi-backend
docker-compose logs -f gradio-app
```

### Step 4: Verify Deployment

```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:4180/ping

# Access application
# Open browser to http://localhost:4180
```

---

## Kubernetes Deployment

### Step 1: Prepare Cluster

```bash
# Create namespace
kubectl create namespace slp-simucase

# Install cert-manager (for SSL)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install nginx ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
```

### Step 2: Configure Secrets

```bash
# Create secrets from .env file
kubectl create secret generic slp-secrets \
  --from-literal=AUTH0_DOMAIN=$AUTH0_DOMAIN \
  --from-literal=AUTH0_CLIENT_ID=$AUTH0_CLIENT_ID \
  --from-literal=AUTH0_CLIENT_SECRET=$AUTH0_CLIENT_SECRET \
  --from-literal=AUTH0_API_AUDIENCE=$AUTH0_API_AUDIENCE \
  --from-literal=AUTH0_ISSUER_URL=$AUTH0_ISSUER_URL \
  --from-literal=OAUTH2_REDIRECT_URL=$OAUTH2_REDIRECT_URL \
  --from-literal=OAUTH2_PROXY_COOKIE_SECRET=$OAUTH2_PROXY_COOKIE_SECRET \
  --from-literal=DATABASE_URL=$DATABASE_URL \
  --from-literal=POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  --from-literal=REDIS_URL=$REDIS_URL \
  --from-literal=REDIS_PASSWORD=$REDIS_PASSWORD \
  --from-literal=OPENAI_API_KEY=$OPENAI_API_KEY \
  --from-literal=GOOGLE_API_KEY=$GOOGLE_API_KEY \
  --from-literal=ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -n slp-simucase
```

### Step 3: Build and Push Images

```bash
# Build images
docker build -t your-registry/slp-gradio:latest -f Dockerfile .
docker build -t your-registry/slp-fastapi:latest -f Dockerfile.fastapi .

# Push to registry
docker push your-registry/slp-gradio:latest
docker push your-registry/slp-fastapi:latest
```

### Step 4: Update Kubernetes Manifests

Edit `k8s/deployment.yaml`:
- Replace `your-registry` with your container registry
- Replace `your-domain.com` with your actual domain
- Update resource limits as needed

### Step 5: Deploy

```bash
# Apply manifests
kubectl apply -f k8s/ -n slp-simucase

# Watch deployment
kubectl get pods -n slp-simucase -w

# Check ingress
kubectl get ingress -n slp-simucase
```

### Step 6: Verify

```bash
# Check pod status
kubectl get pods -n slp-simucase

# Check services
kubectl get svc -n slp-simucase

# View logs
kubectl logs -f deployment/gradio-app -n slp-simucase
kubectl logs -f deployment/fastapi-backend -n slp-simucase

# Port forward for testing
kubectl port-forward svc/oauth2-proxy 4180:4180 -n slp-simucase
```

---

## SSL/TLS Configuration

### Option 1: Let's Encrypt with cert-manager

```bash
# Create ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@domain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

The ingress in `k8s/deployment.yaml` is already configured to use this issuer.

### Option 2: Custom Certificate

```bash
# Create TLS secret
kubectl create secret tls slp-tls-cert \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  -n slp-simucase
```

---

## Database Setup

### Migrations

```bash
# Install Alembic
pip install alembic

# Initialize (first time only)
cd backend
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

### Backup

```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres slp_rbac > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
docker-compose exec -T postgres psql -U postgres slp_rbac < backup_20240101_120000.sql
```

---

## Monitoring Setup

### Prometheus + Grafana

```yaml
# Add to docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

---

## Backup Strategy

### Automated Backups

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/backups

# Database backup
docker-compose exec -T postgres pg_dump -U postgres slp_rbac | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Generated files backup
tar -czf $BACKUP_DIR/files_$DATE.tar.gz generated_case_files/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -mtime +30 -delete
EOF

chmod +x backup.sh

# Add to crontab (daily at 2 AM)
0 2 * * * /path/to/backup.sh
```

---

## Troubleshooting

### Common Issues

**OAuth2 Callback Error**
```
Solution: Verify OAUTH2_REDIRECT_URL matches Auth0 allowed callback URLs exactly
```

**Database Connection Failed**
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Verify connection string
docker-compose exec fastapi-backend env | grep DATABASE_URL
```

**Container Crashes**
```bash
# Check logs
docker-compose logs <service-name>

# Check resource usage
docker stats
```

**Permission Denied**
```
Solution: Check Auth0 role assignments and API permissions
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=debug
docker-compose up -d
docker-compose logs -f
```

---

## Production Checklist

- [ ] Auth0 configured with production settings
- [ ] SSL/TLS certificates installed
- [ ] Environment variables secured (no defaults)
- [ ] Database backups automated
- [ ] Monitoring and alerting configured
- [ ] Log aggregation set up
- [ ] Resource limits configured
- [ ] Rate limiting enabled
- [ ] Security headers configured
- [ ] Audit logging tested
- [ ] Disaster recovery plan documented
- [ ] Team access documented

---

**Next Steps:**
- Review [RBAC Guide](rbac-guide.md)
- Set up [Monitoring](monitoring.md)
- Configure [Backups](backup-recovery.md)
