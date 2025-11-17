# AI Gameplay Bot - API Documentation

Complete API reference for all services.

## Table of Contents

1. [Prediction Services](#prediction-services)
   - [Neural Network API](#neural-network-api)
   - [Transformer API](#transformer-api)
2. [Control Backend API](#control-backend-api)
3. [Error Handling](#error-handling)
4. [Rate Limiting](#rate-limiting)

---

## Prediction Services

### Neural Network API

**Base URL:** `http://localhost:5000`

#### POST /predict

Predict an action based on the current game state using the Neural Network model.

**Request:**

```http
POST /predict HTTP/1.1
Content-Type: application/json

{
  "state": [0.1, 0.2, ..., 0.9]  // Array of 128 float values
}
```

**Request Body Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| state | array[float] | Yes | Game state features (128 dimensions) |

**Response:**

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "action": "move_forward"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| action | string | Predicted action name |

**Possible Actions:**
- `move_forward`
- `move_backward`
- `turn_left`
- `turn_right`
- `attack`
- `jump`
- `interact`
- `use_item`
- `open_inventory`
- `cast_spell`

**Example:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"state": [0.5, 0.3, ..., 0.7]}'
```

```python
import requests

state = [0.5] * 128
response = requests.post(
    "http://localhost:5000/predict",
    json={"state": state}
)
action = response.json()["action"]
print(f"Predicted action: {action}")
```

**Error Responses:**

```http
HTTP/1.1 400 Bad Request
{
  "error": "Missing 'state' in request"
}
```

```http
HTTP/1.1 400 Bad Request
{
  "error": "'state' must be a list of length 128"
}
```

```http
HTTP/1.1 500 Internal Server Error
{
  "error": "Model prediction failed: <error message>"
}
```

#### GET /health

Check if the Neural Network service is running and healthy.

**Response:**

```http
HTTP/1.1 200 OK
{
  "status": "healthy",
  "service": "neural_network"
}
```

---

### Transformer API

**Base URL:** `http://localhost:5001`

#### POST /predict

Predict an action using the Transformer model (considers sequence context).

**Request:**

```http
POST /predict HTTP/1.1
Content-Type: application/json

{
  "state": [0.1, 0.2, ..., 0.9]  // Array of 128 float values
}
```

**Note:** While the request format is the same as the NN API, the Transformer model internally maintains sequence history for better context-aware predictions.

**Response:**

```http
HTTP/1.1 200 OK
{
  "action": "attack"
}
```

**Example:**

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"state": [0.5, 0.3, ..., 0.7]}'
```

#### GET /health

Check if the Transformer service is running and healthy.

**Response:**

```http
HTTP/1.1 200 OK
{
  "status": "healthy",
  "service": "transformer"
}
```

---

## Control Backend API

**Base URL:** `http://localhost:8000`

### GET /api/status

Get the current status of all services and the active model.

**Response:**

```http
HTTP/1.1 200 OK
{
  "nn_running": true,
  "transformer_running": false,
  "active_model": "nn",
  "timestamp": 1234567890.123
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| nn_running | boolean | True if NN service is running |
| transformer_running | boolean | True if Transformer service is running |
| active_model | string | Currently active model ("nn" or "transformer") |
| timestamp | float | Unix timestamp of the status check |

**Example:**

```bash
curl http://localhost:8000/api/status
```

---

### POST /api/start_nn

Start the Neural Network prediction service.

**Request:**

```http
POST /api/start_nn HTTP/1.1
```

**Response:**

```http
HTTP/1.1 200 OK
{
  "success": true,
  "message": "NN service started"
}
```

**Error Response:**

```http
HTTP/1.1 500 Internal Server Error
{
  "success": false,
  "message": "Failed to start NN service"
}
```

---

### POST /api/stop_nn

Stop the Neural Network prediction service.

**Request:**

```http
POST /api/stop_nn HTTP/1.1
```

**Response:**

```http
HTTP/1.1 200 OK
{
  "success": true,
  "message": "NN service stopped"
}
```

---

### POST /api/start_transformer

Start the Transformer prediction service.

**Request:**

```http
POST /api/start_transformer HTTP/1.1
```

**Response:**

```http
HTTP/1.1 200 OK
{
  "success": true,
  "message": "Transformer service started"
}
```

---

### POST /api/stop_transformer

Stop the Transformer prediction service.

**Request:**

```http
POST /api/stop_transformer HTTP/1.1
```

**Response:**

```http
HTTP/1.1 200 OK
{
  "success": true,
  "message": "Transformer service stopped"
}
```

---

### POST /api/set_active_model

Set which model should be used for predictions.

**Request:**

```http
POST /api/set_active_model HTTP/1.1
Content-Type: application/json

{
  "model": "nn"
}
```

**Request Body:**

| Parameter | Type | Required | Values | Description |
|-----------|------|----------|--------|-------------|
| model | string | Yes | "nn" or "transformer" | Model to activate |

**Response:**

```http
HTTP/1.1 200 OK
{
  "success": true,
  "active_model": "nn",
  "message": "Active model set to nn"
}
```

**Error Response:**

```http
HTTP/1.1 400 Bad Request
{
  "success": false,
  "message": "Invalid model. Must be 'nn' or 'transformer'"
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/api/set_active_model \
  -H "Content-Type: application/json" \
  -d '{"model": "transformer"}'
```

---

### POST /api/test_predict

Test the prediction service with a random state.

**Request:**

```http
POST /api/test_predict HTTP/1.1
Content-Type: application/json

{
  "model": "nn"
}
```

**Request Body:**

| Parameter | Type | Required | Values | Description |
|-----------|------|----------|--------|-------------|
| model | string | Yes | "nn" or "transformer" | Model to test |

**Response:**

```http
HTTP/1.1 200 OK
{
  "success": true,
  "action": "move_forward",
  "model": "nn"
}
```

**Error Response:**

```http
HTTP/1.1 500 Internal Server Error
{
  "success": false,
  "message": "Failed to connect to nn service. Is it running?"
}
```

---

### GET /health

Check if the Control Backend is running.

**Response:**

```http
HTTP/1.1 200 OK
{
  "status": "healthy",
  "timestamp": 1234567890.123
}
```

---

## Error Handling

All APIs use standard HTTP status codes and return JSON error messages.

### Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Endpoint not found |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service is down or not responding |

### Error Response Format

```json
{
  "error": "Description of the error",
  "details": "Additional error details (optional)"
}
```

### Common Errors

#### 1. Invalid Input Size

```json
{
  "error": "'state' must be a list of length 128"
}
```

**Cause:** Input state array doesn't have exactly 128 elements.

**Solution:** Ensure your state vector has 128 float values.

#### 2. Missing Parameter

```json
{
  "error": "Missing 'state' in request"
}
```

**Cause:** Required parameter not provided in request body.

**Solution:** Include all required parameters.

#### 3. Service Not Running

```json
{
  "success": false,
  "message": "Failed to connect to nn service. Is it running?"
}
```

**Cause:** Prediction service is not running or not accessible.

**Solution:** Start the service using the control backend or manually.

---

## Rate Limiting

Currently, no rate limiting is enforced. For production deployments:

**Recommendations:**
- Implement rate limiting at the API gateway level
- Use Redis for distributed rate limiting
- Typical limits: 100 requests/minute per client
- Monitor and adjust based on server capacity

**Example (using Flask-Limiter):**

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per minute"]
)

@app.route("/predict", methods=["POST"])
@limiter.limit("100 per minute")
def predict():
    # ... prediction logic
```

---

## WebSocket Support (Future)

For real-time, low-latency gameplay, WebSocket support is planned:

```javascript
const ws = new WebSocket('ws://localhost:5000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'predict',
    state: [0.5, 0.3, ...]
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Action:', data.action);
};
```

---

## Authentication (Future)

For production deployments, implement API key authentication:

```http
POST /predict HTTP/1.1
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "state": [0.5, 0.3, ..., 0.7]
}
```

---

## SDK Examples

### Python

```python
import requests

class GameplayBotClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url

    def predict(self, state):
        response = requests.post(
            f"{self.base_url}/predict",
            json={"state": state}
        )
        response.raise_for_status()
        return response.json()["action"]

    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = GameplayBotClient()
state = [0.5] * 128
action = client.predict(state)
print(f"Action: {action}")
```

### JavaScript

```javascript
class GameplayBotClient {
  constructor(baseUrl = 'http://localhost:5000') {
    this.baseUrl = baseUrl;
  }

  async predict(state) {
    const response = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.action;
  }

  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/health`);
    return await response.json();
  }
}

// Usage
const client = new GameplayBotClient();
const state = Array(128).fill(0.5);
const action = await client.predict(state);
console.log('Action:', action);
```

---

## Support

For API issues or questions:
- GitHub Issues: https://github.com/your-username/ai-gameplay-bot/issues
- Documentation: See `SETUP.md` and `README.md`
