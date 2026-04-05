#!/bin/bash
# Run this on your local machine to build and test Docker
# Usage: bash scripts/docker_test.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Building Docker image ==="
docker build -t veritas-ai:latest -f server/Dockerfile .
echo "Build successful!"

echo ""
echo "=== Step 2: Running container ==="
docker run -d -p 8000:8000 --name veritas-test veritas-ai:latest
echo "Container started. Waiting 5 seconds for startup..."
sleep 5

echo ""
echo "=== Step 3: Testing endpoints ==="
echo "Health check:"
curl -s http://localhost:8000/health
echo ""

echo "Schema check:"
curl -s http://localhost:8000/schema | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK — keys:', list(d.keys()))"

echo ""
echo "Reset check:"
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK — case_id:', d.get('case_id','?'))"

echo ""
echo "=== All checks passed! ==="
echo "Open http://localhost:8000/docs in your browser to try the Swagger UI."
echo ""
echo "To stop the container: docker stop veritas-test && docker rm veritas-test"
