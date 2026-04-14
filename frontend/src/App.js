import React, { useEffect, useState } from "react";
import "./App.css";

const API_BASE = process.env.REACT_APP_API_URL || "http://192.168.8.200:5000";
const PAGE_SIZE = 8;
const REFRESH_INTERVAL_MS = 30000;
const USE_MOCK_DATA = false;

const MOCK_DETECTIONS = [
  {
    id: 108,
    timestamp: "2026-04-11T21:32:00-03:00",
    device_id: "pi5-unit-01",
    detected: true,
    night_mode: true,
    image_path: "",
    detections: [
      { species: "deer", confidence: 0.94, bbox: [115, 84, 311, 352] },
      { species: "rabbit", confidence: 0.83, bbox: [356, 240, 430, 318] },
    ],
    species_count: { deer: 1, rabbit: 1 },
    summary: "2 animals detected: 1 deer, 1 rabbit",
  },
  {
    id: 107,
    timestamp: "2026-04-11T21:18:00-03:00",
    device_id: "pi5-unit-01",
    detected: true,
    night_mode: true,
    image_path: "",
    detections: [{ species: "bear", confidence: 0.91, bbox: [130, 92, 338, 366] }],
    species_count: { bear: 1 },
    summary: "1 animal detected: 1 bear",
  },
  {
    id: 106,
    timestamp: "2026-04-11T20:56:00-03:00",
    device_id: "pi5-unit-02",
    detected: true,
    night_mode: false,
    image_path: "",
    detections: [
      { species: "deer", confidence: 0.88, bbox: [122, 88, 296, 341] },
      { species: "deer", confidence: 0.86, bbox: [312, 96, 468, 338] },
    ],
    species_count: { deer: 2 },
    summary: "2 animals detected: 2 deer",
  },
  {
    id: 105,
    timestamp: "2026-04-11T20:22:00-03:00",
    device_id: "pi5-unit-03",
    detected: true,
    night_mode: false,
    image_path: "",
    detections: [{ species: "fox", confidence: 0.79, bbox: [177, 126, 298, 287] }],
    species_count: { fox: 1 },
    summary: "1 animal detected: 1 fox",
  },
  {
    id: 104,
    timestamp: "2026-04-11T19:47:00-03:00",
    device_id: "pi5-unit-01",
    detected: true,
    night_mode: false,
    image_path: "",
    detections: [{ species: "rabbit", confidence: 0.84, bbox: [235, 198, 315, 282] }],
    species_count: { rabbit: 1 },
    summary: "1 animal detected: 1 rabbit",
  },
  {
    id: 103,
    timestamp: "2026-04-11T18:54:00-03:00",
    device_id: "pi5-unit-02",
    detected: true,
    night_mode: false,
    image_path: "",
    detections: [{ species: "moose", confidence: 0.89, bbox: [102, 74, 368, 381] }],
    species_count: { moose: 1 },
    summary: "1 animal detected: 1 moose",
  },
  {
    id: 102,
    timestamp: "2026-04-11T18:11:00-03:00",
    device_id: "pi5-unit-03",
    detected: true,
    night_mode: false,
    image_path: "",
    detections: [
      { species: "deer", confidence: 0.82, bbox: [118, 90, 294, 334] },
      { species: "deer", confidence: 0.81, bbox: [320, 115, 470, 342] },
      { species: "rabbit", confidence: 0.75, bbox: [408, 257, 462, 309] },
    ],
    species_count: { deer: 2, rabbit: 1 },
    summary: "3 animals detected: 2 deer, 1 rabbit",
  },
  {
    id: 101,
    timestamp: "2026-04-11T17:39:00-03:00",
    device_id: "pi5-unit-01",
    detected: true,
    night_mode: false,
    image_path: "",
    detections: [{ species: "bear", confidence: 0.87, bbox: [140, 104, 348, 360] }],
    species_count: { bear: 1 },
    summary: "1 animal detected: 1 bear",
  },
];

const MOCK_STATS = {
  total_detections: 124,
  species_breakdown: {
    deer: 62,
    bear: 18,
    rabbit: 21,
    fox: 9,
    moose: 14,
  },
};

function App() {
  const [detections, setDetections] = useState([]);
  const [stats, setStats] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");
  const [streamStatus, setStreamStatus] = useState("idle");
  const [activeNav, setActiveNav] = useState("dashboard");
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(false);

  useEffect(() => {
    let isMounted = true;

    if (USE_MOCK_DATA) {
      const pageStart = page * PAGE_SIZE;
      const pageSlice = MOCK_DETECTIONS.slice(pageStart, pageStart + PAGE_SIZE);
      setStats(MOCK_STATS);
      setDetections(pageSlice);
      setHasMore(pageStart + PAGE_SIZE < MOCK_DETECTIONS.length);
      setApiStatus("connected");
      setStreamStatus("live");
      setError("");
      setLoading(false);
      return undefined;
    }

    async function loadDashboardData(showLoading = false) {
      if (showLoading && isMounted) {
        setLoading(true);
      }

      try {
        const [statsResponse, detectionsResponse] = await Promise.all([
          fetch(`${API_BASE}/stats`),
          fetch(
            `${API_BASE}/detections?limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}`
          ),
        ]);

        let healthState = "connected";
        try {
          const healthResponse = await fetch(`${API_BASE}/health`);
          healthState = healthResponse.ok ? "connected" : "limited";
        } catch (healthError) {
          healthState = "limited";
        }

        if (!statsResponse.ok || !detectionsResponse.ok) {
          throw new Error("Unable to load dashboard data from the API.");
        }

        const [statsData, detectionsData] = await Promise.all([
          statsResponse.json(),
          detectionsResponse.json(),
        ]);

        if (!isMounted) {
          return;
        }

        setStats(statsData);
        setDetections(Array.isArray(detectionsData) ? detectionsData : []);
        setHasMore(Array.isArray(detectionsData) && detectionsData.length === PAGE_SIZE);
        setApiStatus(healthState);
        setError("");
      } catch (requestError) {
        if (!isMounted) {
          return;
        }

        setError(requestError.message || "Unable to reach the backend API.");
        setApiStatus("offline");
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    loadDashboardData(true);
    const intervalId = window.setInterval(() => loadDashboardData(false), REFRESH_INTERVAL_MS);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, [page]);

  useEffect(() => {
    if (USE_MOCK_DATA) {
      return undefined;
    }

    if (typeof EventSource === "undefined") {
      setStreamStatus("unsupported");
      return undefined;
    }

    const source = new EventSource(`${API_BASE}/stream`);
    let streamClosed = false;

    source.onopen = () => {
      setStreamStatus("live");
    };

    source.onmessage = (event) => {
      try {
        const incomingDetection = JSON.parse(event.data);
        setDetections((current) => {
          const next = [incomingDetection, ...current.filter((item) => item.id !== incomingDetection.id)];
          return next.slice(0, PAGE_SIZE);
        });
      } catch (parseError) {
        setStreamStatus("fallback");
      }
    };

    source.onerror = () => {
      if (!streamClosed) {
        setStreamStatus("fallback");
        source.close();
        streamClosed = true;
      }
    };

    return () => {
      streamClosed = true;
      source.close();
    };
  }, []);

  const latestDetection = detections[0] || null;

  const speciesBreakdown =
    stats?.species_breakdown && Object.keys(stats.species_breakdown).length > 0
      ? stats.species_breakdown
      : detections.reduce((accumulator, detection) => {
          const counts = detection?.species_count || {};
          Object.entries(counts).forEach(([species, count]) => {
            accumulator[species] = (accumulator[species] || 0) + Number(count || 0);
          });
          return accumulator;
        }, {});

  const statCards = [
    {
      label: "Total Events",
      value: stats?.total_detections ?? detections.length,
      tone: "primary",
    },
    {
      label: "Species Seen",
      value: Object.keys(speciesBreakdown).length,
      tone: "neutral",
    },
    {
      label: "Live Status",
      value: streamStatus === "live" ? "Streaming" : "Polling",
      tone: "neutral",
    },
    {
      label: "Night Mode",
      value: latestDetection?.night_mode ? "On" : "Off",
      tone: "neutral",
    },
  ];

  function handleNavClick(sectionId, navKey) {
    setActiveNav(navKey);

    const target = document.getElementById(sectionId);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  return (
    <div className="dashboard-shell">
      <div className="dashboard-backdrop" />
      <main className="dashboard dashboard-product container-fluid px-2 px-md-3">
        <div className="workspace-shell row g-3 g-xl-4">
          <aside className="col-12 col-xl-3">
            <div className="sidebar h-100">
            <div className="brand-block">
              <div className="brand-mark">W</div>
              <div>
                <strong>WildSight</strong>
                <span>Monitoring Suite</span>
              </div>
            </div>

            <div className="nav-group">
              <p className="nav-label">Menu</p>
              <button
                type="button"
                className={`nav-item ${activeNav === "dashboard" ? "active" : ""}`}
                onClick={() => handleNavClick("dashboard-top", "dashboard")}
              >
                Dashboard
              </button>
              <button
                type="button"
                className={`nav-item ${activeNav === "live-feed" ? "active" : ""}`}
                onClick={() => handleNavClick("latest-detection", "live-feed")}
              >
                Live Feed
              </button>
              <button
                type="button"
                className={`nav-item ${activeNav === "detections" ? "active" : ""}`}
                onClick={() => handleNavClick("detection-history", "detections")}
              >
                Detections
              </button>
              <button
                type="button"
                className={`nav-item ${activeNav === "analytics" ? "active" : ""}`}
                onClick={() => handleNavClick("species-analytics", "analytics")}
              >
                Analytics
              </button>
            </div>

            <div className="nav-group">
              <p className="nav-label">System</p>
              <button
                type="button"
                className={`nav-item ${activeNav === "api-health" ? "active" : ""}`}
                onClick={() => handleNavClick("dashboard-top", "api-health")}
              >
                API Health
              </button>
              <button
                type="button"
                className={`nav-item ${activeNav === "images" ? "active" : ""}`}
                onClick={() => handleNavClick("latest-detection", "images")}
              >
                Images
              </button>
              <button
                type="button"
                className={`nav-item ${activeNav === "settings" ? "active" : ""}`}
                onClick={() => handleNavClick("monitoring-notes", "settings")}
              >
                Settings
              </button>
            </div>

            <div className="sidebar-card">
              <p className="panel-kicker">Current Status</p>
              <h3>Wildlife watch is running</h3>
              <p>Monitor detections, species activity, and evidence snapshots in one place.</p>
            </div>
            </div>
          </aside>

          <section className="workspace-main col-12 col-xl-9">
            <header className="topbar row g-3 align-items-start" id="dashboard-top">
              <div className="col-12 col-lg-7">
                <p className="eyebrow">Wildlife Detection System</p>
                <h1>Dashboard</h1>
              </div>

              <div className="topbar-actions col-12 col-lg-5">
                <StatusPill label="API" value={formatApiStatus(apiStatus)} tone={apiStatus} />
                <StatusPill
                  label="Updates"
                  value={formatStreamStatus(streamStatus)}
                  tone={streamStatus === "live" ? "connected" : "limited"}
                />
              </div>
            </header>

            {error ? (
              <section className="message-panel error-panel">
                <strong>Dashboard warning:</strong> {error}
              </section>
            ) : null}

            <section className="row g-3" id="overview-stats">
              {statCards.map((card) => (
                <div key={card.label} className="col-12 col-sm-6 col-xxl-3">
                  <article className={`stat-card ${card.tone} h-100`}>
                    <span>{card.label}</span>
                    <strong>{card.value}</strong>
                  </article>
                </div>
              ))}
            </section>

            <section className="row g-3">
              <div className="col-12 col-xxl-8">
              <article className="panel featured-panel h-100" id="latest-detection">
                <div className="panel-header">
                  <div>
                    <p className="panel-kicker">Latest Detection</p>
                    <h2>Live Event Snapshot</h2>
                  </div>
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => setSelectedDetection(latestDetection)}
                    disabled={!latestDetection}
                  >
                    View Details
                  </button>
                </div>

                {loading ? (
                  <PanelMessage text="Loading latest wildlife event..." />
                ) : latestDetection ? (
                  <div className="featured-content">
                    <div className="featured-copy">
                      <div className="featured-meta">
                        <span>{formatTimestamp(latestDetection.timestamp)}</span>
                        <span>{latestDetection.device_id || "Unknown device"}</span>
                      </div>
                      <h3>{latestDetection.summary || "Detection event recorded"}</h3>
                      <p>
                        {latestDetection.detected
                          ? "The system recorded at least one wildlife detection in this event."
                          : "The event was logged, but no wildlife was confirmed in the frame."}
                      </p>

                      <div className="species-list">
                        {Object.entries(latestDetection.species_count || {}).length > 0 ? (
                          Object.entries(latestDetection.species_count || {}).map(([species, count]) => (
                            <span key={species} className="species-chip">
                              {species}: {count}
                            </span>
                          ))
                        ) : (
                          <span className="species-chip muted">No species breakdown available</span>
                        )}
                      </div>
                    </div>

                    <div className="featured-image-wrap">
                      {getImageUrl(latestDetection) ? (
                        <img
                          src={getImageUrl(latestDetection)}
                          alt={latestDetection.summary || "Wildlife capture"}
                          className="featured-image"
                        />
                      ) : (
                        <div className="image-fallback">No image available</div>
                      )}
                    </div>
                  </div>
                ) : (
                  <PanelMessage text="No detection events have been logged yet." />
                )}
              </article>
              </div>

              <div className="col-12 col-md-6 col-xxl-4">
              <article className="panel species-panel h-100" id="species-analytics">
                <div className="panel-header">
                  <div>
                    <p className="panel-kicker">Species Breakdown</p>
                    <h2>Observed Animals</h2>
                  </div>
                </div>

                {Object.keys(speciesBreakdown).length > 0 ? (
                  <div className="breakdown-list">
                    {Object.entries(speciesBreakdown)
                      .sort(([, a], [, b]) => b - a)
                      .map(([species, count]) => (
                        <div key={species} className="breakdown-row">
                          <span className="breakdown-label">{species}</span>
                          <div className="breakdown-bar">
                            <div
                              className="breakdown-fill"
                              style={{ width: `${getBarWidth(count, speciesBreakdown)}%` }}
                            />
                          </div>
                          <strong>{count}</strong>
                        </div>
                      ))}
                  </div>
                ) : (
                  <PanelMessage text="Species counts will appear here once detections are available." />
                )}
              </article>
              </div>

              <div className="col-12 col-md-6 col-xxl-4">
              <article className="panel compact-panel h-100" id="monitoring-notes">
                <div className="panel-header">
                  <div>
                    <p className="panel-kicker">Monitoring Notes</p>
                    <h2>Field Summary</h2>
                  </div>
                </div>
                <div className="detail-list">
                  <DetailItem label="Current Page" value={`Page ${page + 1}`} />
                  <DetailItem label="Latest Event" value={latestDetection ? `#${latestDetection.id}` : "None"} />
                  <DetailItem label="Images" value={latestDetection?.image_path ? "Available" : "Pending"} />
                </div>
              </article>
              </div>

              <div className="col-12">
              <article className="panel table-panel history-panel" id="detection-history">
                <div className="panel-header">
                  <div>
                    <p className="panel-kicker">Detection History</p>
                    <h2>Recent Events</h2>
                  </div>

                  <div className="table-actions">
                    <button
                      type="button"
                      className="ghost-button"
                      onClick={() => setPage((current) => Math.max(current - 1, 0))}
                      disabled={page === 0 || loading}
                    >
                      Previous
                    </button>
                    <span className="page-indicator">Page {page + 1}</span>
                    <button
                      type="button"
                      className="ghost-button"
                      onClick={() => setPage((current) => current + 1)}
                      disabled={!hasMore || loading}
                    >
                      Next
                    </button>
                  </div>
                </div>

                <div className="table-wrap">
                  <table className="detection-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Timestamp</th>
                        <th>Summary</th>
                        <th>Species</th>
                        <th>Image</th>
                      </tr>
                    </thead>
                    <tbody>
                      {loading ? (
                        <tr>
                          <td colSpan="5" className="empty-cell">
                            Loading detection history...
                          </td>
                        </tr>
                      ) : detections.length > 0 ? (
                        detections.map((detection) => (
                          <tr key={detection.id} onClick={() => setSelectedDetection(detection)}>
                            <td>{detection.id}</td>
                            <td>{formatTimestamp(detection.timestamp)}</td>
                            <td>{detection.summary || "Detection event"}</td>
                            <td>{formatSpecies(detection.species_count)}</td>
                            <td>{detection.image_path ? "Available" : "None"}</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan="5" className="empty-cell">
                            No detection records to display.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </article>
              </div>
            </section>
          </section>
        </div>

        {selectedDetection ? (
          <div className="modal-backdrop" onClick={() => setSelectedDetection(null)}>
            <div
              className="detail-modal popup-detail-modal"
              onClick={(event) => event.stopPropagation()}
              role="dialog"
              aria-modal="true"
            >
              <div className="panel-header">
                <div>
                  <p className="panel-kicker">Detection Details</p>
                  <h2>Event #{selectedDetection.id}</h2>
                </div>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => setSelectedDetection(null)}
                >
                  Close
                </button>
              </div>

              <div className="detail-grid">
                <div className="detail-list">
                  <DetailItem label="Timestamp" value={formatTimestamp(selectedDetection.timestamp)} />
                  <DetailItem label="Device" value={selectedDetection.device_id || "Unknown"} />
                  <DetailItem
                    label="Night Mode"
                    value={selectedDetection.night_mode ? "Enabled" : "Disabled"}
                  />
                  <DetailItem
                    label="Detected"
                    value={selectedDetection.detected ? "Yes" : "No"}
                  />
                  <DetailItem
                    label="Summary"
                    value={selectedDetection.summary || "No summary available"}
                  />
                </div>

                <div>
                  {getImageUrl(selectedDetection) ? (
                    <img
                      src={getImageUrl(selectedDetection)}
                      alt={selectedDetection.summary || "Detection detail"}
                      className="detail-image"
                    />
                  ) : (
                    <div className="image-fallback detail-fallback">No image available</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : null}

      </main>
    </div>
  );
}

function StatusPill({ label, value, tone }) {
  return (
    <div className={`status-pill ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PanelMessage({ text }) {
  return <div className="panel-message">{text}</div>;
}

function DetailItem({ label, value }) {
  return (
    <div className="detail-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function getImageUrl(detection) {
  if (!detection?.image_path) {
    return "";
  }

  const filename = detection.image_path.split(/[\\/]/).pop();
  return `${API_BASE}/images/${encodeURIComponent(filename)}`;
}

function formatTimestamp(timestamp) {
  if (!timestamp) {
    return "Unknown time";
  }

  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }

  return new Intl.DateTimeFormat("en-CA", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function formatSpecies(speciesCount) {
  const entries = Object.entries(speciesCount || {});
  if (entries.length === 0) {
    return "No data";
  }

  return entries.map(([species, count]) => `${species} (${count})`).join(", ");
}

function getBarWidth(value, breakdown) {
  const values = Object.values(breakdown || {});
  const max = values.length > 0 ? Math.max(...values) : 1;
  return (value / max) * 100;
}

function formatApiStatus(status) {
  if (status === "connected") {
    return "Online";
  }

  if (status === "limited") {
    return "Partial";
  }

  if (status === "offline") {
    return "Offline";
  }

  return "Checking";
}

function formatStreamStatus(status) {
  if (status === "live") {
    return "Live";
  }

  if (status === "unsupported") {
    return "Unsupported";
  }

  if (status === "fallback") {
    return "Polling";
  }

  return "Starting";
}

export default App;
