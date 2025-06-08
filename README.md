# ajsbsd.net

A modern Next.js web application featuring Hugging Face API integration and Docker containerization. This project serves as a personal website and experimentation platform for AI/ML model interactions.

## 🌟 Overview

ajsbsd.net is a Next.js-based web application that demonstrates:
- Integration with Hugging Face Inference API
- Modern UI development practices
- Docker containerization for easy deployment
- Experimentation with various AI/ML models

The project is currently live at [ajsbsd.net](https://ajsbsd.net) and serves as both a personal website and a testing ground for AI model interactions.

## 🚀 Features

- **🤖 AI Integration**: Seamless integration with Hugging Face models
- **🎨 Modern UI**: Clean, responsive design without heavy UI frameworks
- **🐳 Docker Ready**: Fully containerized for easy deployment
- **⚡ Next.js 14+**: Built with the latest Next.js features
- **📱 Responsive**: Mobile-first design approach
- **🔧 TypeScript**: Full type safety throughout the application

## 🛠️ Tech Stack

- **Framework**: Next.js 14+
- **Language**: TypeScript/JavaScript
- **Styling**: CSS Modules / Tailwind CSS
- **AI/ML**: Hugging Face Inference API
- **Containerization**: Docker
- **Deployment**: Docker containers

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- Docker (for containerized deployment)
- Hugging Face API token (optional, for AI features)

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/ajsbsd/ajsbsd.net.git
cd ajsbsd.net
```

2. **Install dependencies**
```bash
npm install
# or
yarn install
```

3. **Set up environment variables**
```bash
cp .env.example .env.local
# Edit .env.local with your configuration
```

4. **Run the development server**
```bash
npm run dev
# or
yarn dev
```

5. **Open your browser**
Navigate to [http://localhost:3000](http://localhost:3000)

### Docker Deployment

#### Build and Run with Docker

```bash
# Build the Docker image
docker build -t ajsbsd.net-nextjs-docker .

# Run the container
docker run -p 127.0.0.1:3000:3000 ajsbsd.net-nextjs-docker
```

#### Using Docker Compose (if available)

```bash
# Start the application
docker-compose up -d

# Stop the application
docker-compose down
```

## 🐳 Docker Configuration

The project includes a multi-stage Dockerfile optimized for production:

### Dockerfile Features
- **Multi-stage build**: Separates build and runtime environments
- **Optimized layers**: Minimal final image size
- **Security**: Non-root user execution
- **Performance**: Efficient caching strategies

### Build Arguments
```bash
# Custom build with arguments
docker build \
  --build-arg NODE_ENV=production \
  --build-arg NEXT_PUBLIC_API_URL=https://api.example.com \
  -t ajsbsd.net .
```

## 🤖 Hugging Face Integration

This project demonstrates integration with Hugging Face models following the [Vercel AI SDK guide](https://sdk.vercel.ai/docs/guides/providers/hugging-face).

### Supported Features
- **Text Generation**: Chat completions and text generation
- **Model Inference**: Direct API calls to HF models
- **Streaming**: Real-time response streaming
- **Error Handling**: Robust error management

### Example Usage

```typescript
// Example API route for HF integration
import { HfInference } from '@huggingface/inference'

const hf = new HfInference(process.env.HUGGING_FACE_API_TOKEN)

export async function POST(request: Request) {
  try {
    const { prompt } = await request.json()
    
    const response = await hf.textGeneration({
      model: 'microsoft/DialoGPT-medium',
      inputs: prompt,
      parameters: {
        max_new_tokens: 100,
        temperature: 0.7,
      }
    })
    
    return Response.json({ response: response.generated_text })
  } catch (error) {
    return Response.json({ error: 'Failed to generate response' }, { status: 500 })
  }
}
```

## 🎨 UI Development

The project focuses on creating a clean, modern UI without relying on heavy component libraries like Radix-UI Theme.

### Design Principles
- **Minimalist**: Clean, uncluttered interface
- **Responsive**: Mobile-first approach
- **Accessible**: WCAG 2.1 compliance
- **Fast**: Optimized loading and interactions

### Styling Approach
```css
/* Example of custom styling approach */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}

.card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
}

@media (max-width: 768px) {
  .container {
    padding: 0 0.5rem;
  }
}
```

## 📁 Project Structure

```
ajsbsd.net/
├── components/          # Reusable UI components
├── pages/              # Next.js pages (if using Pages Router)
├── app/                # Next.js app directory (if using App Router)
├── public/             # Static assets
├── styles/             # Global styles and CSS modules
├── lib/                # Utility functions and configurations
├── hooks/              # Custom React hooks
├── types/              # TypeScript type definitions
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration (if available)
├── .env.example        # Environment variables template
└── next.config.js      # Next.js configuration
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file with the following variables:

```bash
# Hugging Face API
HUGGING_FACE_API_TOKEN=your_hf_token_here

# Application Settings
NEXT_PUBLIC_APP_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:3000/api

# Optional: Analytics, monitoring, etc.
NEXT_PUBLIC_GA_ID=your_ga_id
```

### Next.js Configuration

```javascript
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true, // if using App Router
  },
  images: {
    domains: ['huggingface.co'],
  },
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
}

module.exports = nextConfig
```

## 🚢 Deployment

### Production Deployment

1. **Build the application**
```bash
npm run build
```

2. **Start production server**
```bash
npm start
```

### Docker Production Deployment

```bash
# Build production image
docker build -t ajsbsd.net:latest -f Dockerfile .

# Run in production mode
docker run -d \
  --name ajsbsd-net \
  -p 3000:3000 \
  --env-file .env.production \
  ajsbsd.net:latest
```

### Cloud Deployment Options

- **Vercel**: Optimized for Next.js applications
- **Docker Cloud Providers**: AWS ECS, Google Cloud Run, Azure Container Instances
- **Kubernetes**: For orchestrated deployments
- **Traditional VPS**: Using Docker or PM2

## 🧪 Development Notes

### Experimentation Goals
- Testing HF Inference capabilities on constrained servers
- Alternative to OpenLLM/BentoML for model serving
- UI/UX optimization without heavy component libraries
- Docker optimization for efficient deployments

### Current Limitations
- Server constraints limit certain ML model deployments
- Focus on cloud-based inference rather than local model serving

## 📝 Scripts

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "docker:build": "docker build -t ajsbsd.net .",
    "docker:run": "docker run -p 3000:3000 ajsbsd.net",
    "docker:dev": "docker-compose up --build"
  }
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow TypeScript best practices
- Maintain responsive design principles
- Test Docker builds before submitting PRs
- Document any new environment variables

## 📚 Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index)
- [Vercel AI SDK](https://sdk.vercel.ai/docs)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## 🐛 Troubleshooting

### Common Issues

**Docker Build Failures**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t ajsbsd.net .
```

**HF API Errors**
- Verify your API token is valid
- Check rate limits and model availability
- Ensure proper error handling in your code

**Port Conflicts**
```bash
# Check what's using port 3000
lsof -i :3000

# Use different port
docker run -p 3001:3000 ajsbsd.net
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Live Site**: [ajsbsd.net](https://ajsbsd.net)
- **Repository**: [GitHub](https://github.com/ajsbsd/ajsbsd.net)
- **Issues**: [GitHub Issues](https://github.com/ajsbsd/ajsbsd.net/issues)

---

Built with ❤️ using Next.js and Docker. Last updated: November 2023
