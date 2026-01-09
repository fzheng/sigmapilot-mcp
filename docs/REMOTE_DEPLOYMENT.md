# Remote Deployment Guide

This guide covers deploying SigmaPilot MCP Server to Railway with Auth0 authentication.

Once deployed, your MCP server can be connected from **any MCP-compatible AI platform**:
- **Claude.ai** - Via [Connectors](https://claude.ai/settings/connectors) ([documentation](https://support.claude.com/en/articles/11724452-using-the-connectors-directory-to-extend-claude-s-capabilities))
- **ChatGPT** - Via MCP plugin support
- **Claude Desktop** - Via config file
- **Other AI platforms** - Any service supporting MCP protocol

## Prerequisites

- [Railway account](https://railway.app/) (free tier available)
- [Auth0 account](https://auth0.com/) (free tier available)
- GitHub repository with this code

## Step 1: Auth0 Setup

### 1.1 Create Auth0 Account

1. Go to [auth0.com](https://auth0.com/) and sign up
2. Create a new tenant (e.g., `sigmapilot-mcp`)

### 1.2 Create an API

1. In Auth0 Dashboard, go to **Applications > APIs**
2. Click **+ Create API**
3. Fill in:
   - **Name**: `SigmaPilot MCP API`
   - **Identifier**: `https://sigmapilot-mcp.yourdomain.com` (this becomes your `AUTH0_AUDIENCE`)
   - **Signing Algorithm**: RS256
4. Click **Create**

### 1.3 Create an Application (for testing)

1. Go to **Applications > Applications**
2. Click **+ Create Application**
3. Select **Machine to Machine Applications**
4. Name it: `SigmaPilot MCP Client`
5. Select the API you just created
6. Grant all permissions

### 1.4 Note Your Credentials

From the Auth0 Dashboard, note:
- **Domain**: `your-tenant.auth0.com` (Settings > General)
- **API Identifier**: The identifier you created above

## Step 2: Railway Deployment

### 2.1 Connect Repository

1. Go to [railway.app](https://railway.app/)
2. Click **New Project**
3. Select **Deploy from GitHub repo**
4. Authorize and select your `sigmapilot-mcp` repository

### 2.2 Configure Environment Variables

In Railway dashboard, go to **Variables** and add:

```
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_AUDIENCE=https://sigmapilot-mcp.yourdomain.com
RESOURCE_SERVER_URL=https://your-app.up.railway.app/mcp
```

> **Note**: `RESOURCE_SERVER_URL` will be your Railway public URL + `/mcp`

### 2.3 Deploy

1. Railway will automatically detect the `railway.json` and deploy
2. Once deployed, go to **Settings > Networking**
3. Click **Generate Domain** to get a public URL
4. Update `RESOURCE_SERVER_URL` with this URL

### 2.4 Verify Deployment

Your server should be running at:
```
https://your-app.up.railway.app/mcp
```

## Step 3: Connect to AI Platform

### Option A: Claude.ai (Web) - Recommended

1. Go to [claude.ai/settings/connectors](https://claude.ai/settings/connectors)
2. Click **Add MCP Server**
3. Enter your server URL: `https://your-app.up.railway.app/mcp`
4. When prompted, authenticate with Auth0
5. Start using your MCP tools in Claude.ai conversations

### Option B: ChatGPT

ChatGPT supports MCP servers through its plugin system. Add your server URL when configuring MCP plugins.

### Option C: Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "sigmapilot-remote": {
      "url": "https://your-app.up.railway.app/mcp",
      "transport": "streamable-http"
    }
  }
}
```

## Architecture

```
┌─────────────────┐      HTTPS       ┌─────────────────┐
│   Claude.ai     │ ───────────────► │  Railway Server │
│   ChatGPT       │  + OAuth Token   │  (server.py)    │
│   AI Platforms  │                  └────────┬────────┘
└─────────────────┘                           │
                                              ▼
                                     ┌─────────────────┐
                                     │    Auth0        │
                                     │  Token Verify   │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │  TradingView    │
                                     │  Market APIs    │
                                     └─────────────────┘
```

## Troubleshooting

### "Authentication failed"

1. Verify `AUTH0_DOMAIN` is correct (no `https://` prefix)
2. Check `AUTH0_AUDIENCE` matches your API identifier exactly
3. Ensure the token hasn't expired

### "Server not responding"

1. Check Railway logs: `railway logs`
2. Verify the server is running on port from `$PORT`
3. Ensure `HOST=0.0.0.0` is set

### "No data returned"

1. Rate limiting - wait and retry
2. Check if the exchange/symbol is supported

## Security Notes

- Never commit `.env` file to version control
- Rotate Auth0 client secrets periodically
- Use separate Auth0 tenants for dev/prod
- Monitor Auth0 logs for suspicious activity

## Cost Considerations

### Railway
- Free tier: 500 hours/month, 100GB bandwidth
- Hobby: $5/month for always-on deployments

### Auth0
- Free tier: 7,000 active users/month
- Sufficient for personal/small team use

## Local Development

For local testing without Auth0:

```bash
# HTTP mode without auth (development)
uv run python src/sigmapilot_mcp/server.py streamable-http --port 8000

# HTTP mode with auth (requires AUTH0_DOMAIN and AUTH0_AUDIENCE)
AUTH0_DOMAIN=your-tenant.auth0.com AUTH0_AUDIENCE=https://your-api \
  uv run python src/sigmapilot_mcp/server.py streamable-http --auth
```

The server will run at `http://localhost:8000/mcp` without authentication in development mode.
