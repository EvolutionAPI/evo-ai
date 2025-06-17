[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/evolutionapi-evo-ai-badge.png)](https://mseep.ai/app/evolutionapi-evo-ai)

<h1 align="center">Evo AI - AI Agents Platform</h1>

<div align="center">

[![Whatsapp Group](https://img.shields.io/badge/Group-WhatsApp-%2322BC18)](https://evolution-api.com/whatsapp)
[![Discord Community](https://img.shields.io/badge/Discord-Community-blue)](https://evolution-api.com/discord)
[![Postman Collection](https://img.shields.io/badge/Postman-Collection-orange)](https://evolution-api.com/postman)
[![Documentation](https://img.shields.io/badge/Documentation-Official-green)](https://doc.evolution-api.com)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)
[![Support](https://img.shields.io/badge/Donation-picpay-green)](https://app.picpay.com/user/davidsongomes1998)
[![Sponsors](https://img.shields.io/badge/Github-sponsor-orange)](https://github.com/sponsors/EvolutionAPI)

</div>

## Evo AI - AI Agents Platform

Evo AI is an open-source platform for creating and managing AI agents, enabling integration with different AI models and services.

## 🚀 Overview

The Evo AI platform allows:

- Creation and management of AI agents
- Integration with different language models
- Client management and MCP server configuration
- Custom tools management
- **[Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/)**: Base framework for agent development
- **[CrewAI Support](https://github.com/crewAI/crewAI)**: Alternative framework for agent development (in development)
- JWT authentication with email verification
- **[Agent 2 Agent (A2A) Protocol Support](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)**: Interoperability between AI agents
- **[Workflow Agent with LangGraph](https://www.langchain.com/langgraph)**: Building complex agent workflows
- **Secure API Key Management**: Encrypted storage of API keys
- **Agent Organization**: Folder structure for organizing agents by categories

## 🤖 Agent Types

Evo AI supports different types of agents that can be flexibly combined:

### 1. LLM Agent (Language Model)

Agent based on language models like GPT-4, Claude, etc. Can be configured with tools, MCP servers, and sub-agents.

### 2. A2A Agent (Agent-to-Agent)

Agent that implements Google's A2A protocol for agent interoperability.

### 3. Sequential Agent

Executes a sequence of sub-agents in a specific order.

### 4. Parallel Agent

Executes multiple sub-agents simultaneously.

### 5. Loop Agent

Executes sub-agents in a loop with a defined maximum number of iterations.

### 6. Workflow Agent

Executes sub-agents in a custom workflow defined by a graph structure using LangGraph.

### 7. Task Agent

Executes a specific task using a target agent with structured task instructions.

## 🛠️ Technologies

### Backend
- **FastAPI**: Web framework for building the API
- **SQLAlchemy**: ORM for database interaction
- **PostgreSQL**: Main database
- **Alembic**: Migration system
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server
- **Redis**: Cache and session management
- **JWT**: Secure token authentication
- **SendGrid/SMTP**: Email service for notifications (configurable)
- **Jinja2**: Template engine for email rendering
- **Bcrypt**: Password hashing and security
- **LangGraph**: Framework for building stateful, multi-agent workflows

### Frontend
- **Next.js 15**: React framework with App Router
- **React 18**: User interface library
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: Modern component library
- **React Hook Form**: Form management
- **Zod**: Schema validation
- **ReactFlow**: Node-based visual workflows
- **React Query**: Server state management

## 📊 Langfuse Integration (Tracing & Observability)

Evo AI platform natively supports integration with [Langfuse](https://langfuse.com/) for detailed tracing of agent executions, prompts, model responses, and tool calls, using the OpenTelemetry (OTel) standard.

### How to configure

1. **Set environment variables in your `.env`:**

   ```env
   LANGFUSE_PUBLIC_KEY="pk-lf-..."
   LANGFUSE_SECRET_KEY="sk-lf-..."
   OTEL_EXPORTER_OTLP_ENDPOINT="https://cloud.langfuse.com/api/public/otel"
   ```

2. **View in the Langfuse dashboard**
   - Access your Langfuse dashboard to see real-time traces.

## 🤖 Agent 2 Agent (A2A) Protocol Support

Evo AI implements the Google's Agent 2 Agent (A2A) protocol, enabling seamless communication and interoperability between AI agents.

For more information about the A2A protocol, visit [Google's A2A Protocol Documentation](https://google.github.io/A2A/).

## 📋 Prerequisites

### Backend
- **Python**: 3.10 or higher
- **PostgreSQL**: 13.0 or higher
- **Redis**: 6.0 or higher
- **Git**: For version control
- **Make**: For running Makefile commands

### Frontend
- **Node.js**: 18.0 or higher
- **pnpm**: Package manager (recommended) or npm/yarn

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/EvolutionAPI/evo-ai.git
cd evo-ai
```

### 2. Backend Setup

#### Virtual Environment and Dependencies

```bash
# Create and activate virtual environment
make venv
source venv/bin/activate  # Linux/Mac
# or on Windows: venv\Scripts\activate

# Install development dependencies
make install-dev
```

#### Environment Configuration

```bash
# Copy and configure backend environment
cp .env.example .env
# Edit the .env file with your database, Redis, and other settings
```

#### Database Setup

```bash
# Initialize database and apply migrations
make alembic-upgrade

# Seed initial data (admin user, sample clients, etc.)
make seed-all
```

### 3. Frontend Setup

#### Install Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies using pnpm (recommended)
pnpm install

# Or using npm
# npm install

# Or using yarn
# yarn install
```

#### Frontend Environment Configuration

```bash
# Copy and configure frontend environment
cp .env.example .env
# Edit .env with your API URL (default: http://localhost:8000)
```

The frontend `.env` should contain:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🚀 Running the Application

### Development Mode

#### Start Backend (Terminal 1)
```bash
# From project root
make run
# Backend will be available at http://localhost:8000
```

#### Start Frontend (Terminal 2)
```bash
# From frontend directory
cd frontend
pnpm dev

# Or using npm/yarn
# npm run dev
# yarn dev

# Frontend will be available at http://localhost:3000
```

### Production Mode

#### Backend
```bash
make run-prod    # Production with multiple workers
```

#### Frontend
```bash
cd frontend
pnpm build && pnpm start

# Or using npm/yarn
# npm run build && npm start
# yarn build && yarn start
```

## 🐳 Docker Installation

### Full Stack with Docker Compose

```bash
# Build and start all services (backend + database + redis)
make docker-build
make docker-up

# Initialize database with seed data
make docker-seed
```

### Frontend with Docker

```bash
# From frontend directory
cd frontend

# Build frontend image
docker build -t evo-ai-frontend .

# Run frontend container
docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://localhost:8000 evo-ai-frontend
```

Or using the provided docker-compose:

```bash
# From frontend directory
cd frontend
docker-compose up -d
```

## 🎯 Getting Started

After installation, follow these steps:

1. **Access the Frontend**: Open `http://localhost:3000`
2. **Create Admin Account**: Use the seeded admin credentials or register a new account
3. **Configure MCP Server**: Set up your first MCP server connection
4. **Create Client**: Add a client to organize your agents
5. **Build Your First Agent**: Create and configure your AI agent
6. **Test Agent**: Use the chat interface to interact with your agent

### Default Admin Credentials

After running the seeders, you can login with:
- **Email**: Check the seeder output for the generated admin email
- **Password**: Check the seeder output for the generated password

## 🖥️ API Documentation

The interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 👨‍💻 Development Commands

### Backend Commands
```bash
# Database migrations
make alembic-upgrade            # Update database to latest version
make alembic-revision message="description"  # Create new migration

# Seeders
make seed-all                   # Run all seeders

# Code verification
make lint                       # Verify code with flake8
make format                     # Format code with black
```

### Frontend Commands
```bash
# From frontend directory
cd frontend

# Development
pnpm dev                        # Start development server
pnpm build                      # Build for production
pnpm start                      # Start production server
pnpm lint                       # Run ESLint
```

## 🚀 Configuration

### Backend Configuration (.env file)

Key settings include:

```bash
# Database settings
POSTGRES_CONNECTION_STRING="postgresql://postgres:root@localhost:5432/evo_ai"

# Redis settings
REDIS_HOST="localhost"
REDIS_PORT=6379

# AI Engine configuration
AI_ENGINE="adk"  # Options: "adk" (Google Agent Development Kit) or "crewai" (CrewAI framework)

# JWT settings
JWT_SECRET_KEY="your-jwt-secret-key"

# Email provider configuration
EMAIL_PROVIDER="sendgrid"  # Options: "sendgrid" or "smtp"

# Encryption for API keys
ENCRYPTION_KEY="your-encryption-key"
```

### Frontend Configuration (.env file)

```bash
# API Configuration
NEXT_PUBLIC_API_URL="http://localhost:8000"  # Backend API URL
```

> **Note**: While Google ADK is fully supported, the CrewAI engine option is still under active development. For production environments, it's recommended to use the default "adk" engine.

## 🔐 Authentication

The API uses JWT (JSON Web Token) authentication with:

- User registration and email verification
- Login to obtain JWT tokens
- Password recovery flow
- Account lockout after multiple failed login attempts

## 🚀 Star Us on GitHub

If you find EvoAI useful, please consider giving us a star! Your support helps us grow our community and continue improving the product.

[![Star History Chart](https://api.star-history.com/svg?repos=EvolutionAPI/evo-ai&type=Date)](https://www.star-history.com/#EvolutionAPI/evo-ai&Date)

## 🤝 Contributing

We welcome contributions from the community! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the [Apache License 2.0](./LICENSE).
