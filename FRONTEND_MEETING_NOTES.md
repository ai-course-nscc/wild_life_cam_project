# Wildlife Dashboard Frontend Status

## Project Context
This frontend is the dashboard for the Wildlife Detection System. It is based on the project proposal and the `DASHBOARD_API.docx` document shared by the team.

The frontend is intended to:
- show latest wildlife detections
- display detection history
- show summary statistics
- display detection images
- connect to the FastAPI backend

## Current Frontend Status

### Completed
- React dashboard UI is built
- Bootstrap is installed and the current layout uses Bootstrap for responsive structure
- custom CSS is used for the final cream/brown product-style design
- left sidebar navigation is working
- latest detection panel is working
- species breakdown panel is working
- monitoring notes panel is working
- recent detections table is working
- `View Details` opens a popup detail modal
- mock/demo data mode is enabled for presentation
- frontend builds successfully with `npm run build`

### Demo Mode
For presentation, the dashboard is currently using mock data instead of live API data.

Current demo switch in `frontend/src/App.js`:
```js
const USE_MOCK_DATA = true;
```

When backend integration is ready, this can be changed to:
```js
const USE_MOCK_DATA = false;
```

Demo mode currently allows:
- full dashboard rendering without backend dependency
- realistic sample detections
- working navigation
- working popup detail view
- working pagination in the table
- stable demo for meeting/presentation use

## What I Have Built

### UI Sections
- sidebar menu
- dashboard header
- API and update status cards
- summary statistic cards
- latest detection section
- species breakdown section
- monitoring notes section
- recent detections table
- detection popup modal

### Responsiveness
- Bootstrap is included and used for responsive layout
- the dashboard reflows across desktop/tablet/mobile sizes
- custom CSS is used on top of Bootstrap for the final visual design

### Interaction
- sidebar buttons scroll to sections
- `View Details` opens a popup modal for the selected event
- clicking a table row also opens detection details
- pagination buttons move through mock detection records

## What Matches the API Document

The current frontend structure matches the intended frontend direction from the team document:
- React dashboard
- latest detection view
- historical detections view
- stats area
- image area
- detection detail view
- live/update status area

## What Is Still Pending

The main unfinished part is real backend integration.

### Frontend work still pending
- replace mock data with live API data
- test real image rendering from backend
- connect real `/health` response
- connect real `/stream` live updates
- connect `/detections/recent` if charts/analytics are added
- final end-to-end testing with backend

## What Is Needed From Backend Team

The frontend needs the backend to match the documented API contract.

### Required endpoints
- `GET /health`
- `GET /detections?limit=&offset=`
- `GET /detections/recent?hours=`
- `GET /detections/{id}`
- `GET /stats`
- `GET /images/{filename}`
- `GET /stream`

### Detection object shape needed by frontend
```json
{
  "id": 42,
  "timestamp": "2026-04-06T18:30:00-03:00",
  "device_id": "pi5-unit-01",
  "detected": true,
  "night_mode": false,
  "image_path": "captures/2026-04-06_18-30-00.jpg",
  "detections": [
    {
      "species": "deer",
      "confidence": 0.91,
      "bbox": [120.0, 80.0, 300.0, 350.0]
    }
  ],
  "species_count": {
    "deer": 1
  },
  "summary": "1 animal(s) detected: 1 deer"
}
```

### Stats response needed by frontend
```json
{
  "total_detections": 124,
  "species_breakdown": {
    "deer": 89,
    "bear": 22,
    "rabbit": 13
  }
}
```

### Health response needed by frontend
```json
{
  "status": "ok",
  "model": "RF-DETR (ONNX Runtime)",
  "api_version": "2.0.0",
  "timestamp": "2026-04-06T18:30:00-03:00"
}
```

### Image handling needed
- each detection should include `image_path`
- frontend will extract the filename from `image_path`
- backend should serve the image through `GET /images/{filename}`

### Live updates needed
For live updates, the frontend expects:
- endpoint: `GET /stream`
- response type: `text/event-stream`
- payload: full detection object

## Current Backend Gap In Repo

From the current backend code in the repo, these already exist:
- `/detections`
- `/detections/{id}`
- `/stats`
- `/images/{filename}`

These documented endpoints are still missing from the current repo backend:
- `/health`
- `/detections/recent`
- `/stream`

Also, `/stats` may still need `species_breakdown` if that is not already returned.

## Suggested Message For Backend Team

I completed the frontend dashboard UI and demo mode. To finish real integration, I need the backend to match the API document. Please provide or implement:

- `GET /health`
- `GET /detections/recent`
- `GET /stream`
- `species_breakdown` in `/stats`
- full detection object fields: `id`, `timestamp`, `device_id`, `detected`, `night_mode`, `image_path`, `detections`, `species_count`, and `summary`
- working image serving through `/images/{filename}`

Once that is ready, I can switch the frontend from mock data to live backend data and complete final integration.

## Meeting Summary

### Done
- frontend dashboard built
- Bootstrap-based responsive layout added
- custom presentation styling completed
- sidebar navigation working
- popup detail modal working
- mock demo data working
- build verified

### Pending
- real backend connection
- real image integration
- SSE live stream integration
- recent detections API for analytics
- final end-to-end testing
