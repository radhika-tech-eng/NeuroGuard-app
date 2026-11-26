
import { useMemo, useState } from "react";
import { Bell, Search, X, ArrowUpRight, CheckCheck } from "lucide-react";
import "./HeaderActions.css";

const SECTIONS = [
  { label: "Overview", id: "overview", keywords: "home dashboard metrics" },
  { label: "EEG Viewer", id: "eeg-viewer", keywords: "signal waveform channel" },
  { label: "Model Analytics", id: "model-analytics", keywords: "xgboost cnn lstm scores models" },
  { label: "Event Timeline", id: "event-timeline", keywords: "windows events research flags" },
  { label: "Review & Settings", id: "review-settings", keywords: "settings configuration" },
];

export default function HeaderActions({ onNavigate, notifications = [] }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [notificationOpen, setNotificationOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [readIds, setReadIds] = useState([]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return SECTIONS;

    return SECTIONS.filter((item) =>
      `${item.label} ${item.keywords}`.toLowerCase().includes(q)
    );
  }, [query]);

  const unreadCount = notifications.filter(
    (item) => !readIds.includes(item.id)
  ).length;

  function navigate(item) {
    if (onNavigate) {
      onNavigate(item.id);
    } else {
      document.getElementById(item.id)?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }

    setSearchOpen(false);
    setQuery("");
  }

  function markAllRead() {
    setReadIds(notifications.map((item) => item.id));
  }

  return (
    <div className="ha-actions">
      <div className="ha-anchor">
        <button
          type="button"
          className={`ha-icon-button ${searchOpen ? "ha-active" : ""}`}
          aria-label="Search dashboard"
          aria-expanded={searchOpen}
          onClick={() => {
            setSearchOpen((v) => !v);
            setNotificationOpen(false);
          }}
        >
          {searchOpen ? <X size={19} /> : <Search size={19} />}
        </button>

        {searchOpen && (
          <div className="ha-popover ha-search-panel">
            <div className="ha-panel-heading">
              <div>
                <strong>Search workspace</strong>
                <p>Find a dashboard section</p>
              </div>
              <span className="ha-shortcut">ESC</span>
            </div>

            <div className="ha-input-wrap">
              <Search size={17} />
              <input
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Try “EEG” or “models”..."
                onKeyDown={(e) => {
                  if (e.key === "Escape") setSearchOpen(false);
                  if (e.key === "Enter" && matches[0]) navigate(matches[0]);
                }}
              />
            </div>

            <div className="ha-result-label">
              {query ? "SEARCH RESULTS" : "QUICK NAVIGATION"}
            </div>

            <div className="ha-results">
              {matches.length ? (
                matches.map((item) => (
                  <button
                    type="button"
                    key={item.id}
                    className="ha-result"
                    onClick={() => navigate(item)}
                  >
                    <span className="ha-result-icon">
                      <Search size={15} />
                    </span>
                    <span className="ha-result-copy">
                      <strong>{item.label}</strong>
                      <small>{item.keywords}</small>
                    </span>
                    <ArrowUpRight size={15} />
                  </button>
                ))
              ) : (
                <div className="ha-empty">
                  <Search size={22} />
                  <strong>No sections found</strong>
                  <span>Try a different keyword.</span>
                </div>
              )}
            </div>

            <div className="ha-panel-footer">
              <span>NEUROGUARD SEARCH</span>
              <span>{matches.length} results</span>
            </div>
          </div>
        )}
      </div>

      <div className="ha-anchor">
        <button
          type="button"
          className={`ha-icon-button ${notificationOpen ? "ha-active" : ""}`}
          aria-label="Notifications"
          aria-expanded={notificationOpen}
          onClick={() => {
            setNotificationOpen((v) => !v);
            setSearchOpen(false);
          }}
        >
          <Bell size={19} />
          {unreadCount > 0 && (
            <span className="ha-badge">
              {unreadCount > 9 ? "9+" : unreadCount}
            </span>
          )}
        </button>

        {notificationOpen && (
          <div className="ha-popover ha-notification-panel">
            <div className="ha-panel-heading">
              <div>
                <strong>Notifications</strong>
                <p>Updates from this session</p>
              </div>
              <button
                type="button"
                className="ha-read-button"
                onClick={markAllRead}
                disabled={unreadCount === 0}
              >
                <CheckCheck size={14} />
                Mark read
              </button>
            </div>

            <div className="ha-notification-list">
              {notifications.length ? (
                notifications.map((item) => {
                  const isRead = readIds.includes(item.id);

                  return (
                    <button
                      type="button"
                      key={item.id}
                      className={`ha-notification ${isRead ? "ha-read" : ""}`}
                      onClick={() => {
                        setReadIds((previous) =>
                          previous.includes(item.id)
                            ? previous
                            : [...previous, item.id]
                        );

                        if (item.section) {
                          navigate({
                            id: item.section,
                            label: item.title,
                          });
                        }
                      }}
                    >
                      <span className={`ha-notification-dot ${item.type || "info"}`} />
                      <span className="ha-notification-copy">
                        <strong>{item.title}</strong>
                        <span>{item.message}</span>
                        <small>{item.time || "This session"}</small>
                      </span>
                      {!isRead && <span className="ha-unread-mark" />}
                    </button>
                  );
                })
              ) : (
                <div className="ha-empty">
                  <span className="ha-empty-icon"><Bell size={23} /></span>
                  <strong>You're all caught up</strong>
                  <span>Upload an EEG file to receive analysis updates here.</span>
                </div>
              )}
            </div>

            <div className="ha-panel-footer">
              <span>SESSION ACTIVITY</span>
              <span>{notifications.length} updates</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
